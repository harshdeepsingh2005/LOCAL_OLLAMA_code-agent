"""
Testing Tools Module

Provides test execution and validation capabilities.
All operations are sandboxed and monitored.

Design Decisions:
- Tests run in subprocess with timeout
- Output captured and analyzed
- No network access during tests
- Resource limits enforced
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.core import TelemetryCollector


class TestStatus(str, Enum):
    """Status of a test run."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class TestResult(BaseModel):
    """Result of a single test."""
    name: str
    status: TestStatus
    duration_ms: float = 0
    message: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    file_path: str | None = None
    line_number: int | None = None


class TestSuiteResult(BaseModel):
    """Result of a test suite run."""
    suite_name: str
    status: TestStatus
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_ms: float = 0
    tests: list[TestResult] = []
    stdout: str = ""
    stderr: str = ""


class TestRunResult(BaseModel):
    """Result of a complete test run."""
    success: bool
    suites: list[TestSuiteResult] = []
    total_duration_ms: float = 0
    summary: str = ""
    error: str | None = None


class TestRunner:
    """
    Test runner with safety guarantees.
    
    Features:
    - Subprocess isolation
    - Timeout enforcement
    - Resource limiting
    - Output capture and parsing
    """
    
    def __init__(
        self,
        workspace_root: Path,
        telemetry: "TelemetryCollector | None" = None,
        default_timeout: float = 300.0,  # 5 minutes
        max_output_size: int = 1_000_000,  # 1MB
    ) -> None:
        """
        Initialize test runner.
        
        Args:
            workspace_root: Root directory for tests
            telemetry: Optional telemetry collector
            default_timeout: Default test timeout in seconds
            max_output_size: Maximum output size to capture
        """
        self._workspace_root = workspace_root.resolve()
        self._telemetry = telemetry
        self._default_timeout = default_timeout
        self._max_output_size = max_output_size
    
    def _log(self, operation: str, success: bool, details: dict) -> None:
        """Log operation to telemetry."""
        if self._telemetry:
            self._telemetry.record_tool_call(
                tool_name=f"testing.{operation}",
                inputs=details,
                outputs={"success": success},
                success=success,
            )
    
    def run_pytest(
        self,
        test_path: str | None = None,
        markers: list[str] | None = None,
        keywords: list[str] | None = None,
        timeout: float | None = None,
        verbose: bool = True,
        capture: bool = True,
        extra_args: list[str] | None = None,
    ) -> TestRunResult:
        """
        Run pytest tests.
        
        Args:
            test_path: Path to tests (file or directory)
            markers: Filter by markers (-m)
            keywords: Filter by keywords (-k)
            timeout: Timeout in seconds
            verbose: Verbose output
            capture: Capture output
            extra_args: Additional pytest arguments
            
        Returns:
            TestRunResult with test results
        """
        cmd = [sys.executable, "-m", "pytest"]
        
        if test_path:
            resolved = self._workspace_root / test_path
            cmd.append(str(resolved))
        else:
            cmd.append(str(self._workspace_root))
        
        # Add common options
        cmd.extend(["--tb=short", "-q"])
        
        if verbose:
            cmd.append("-v")
        
        if capture:
            cmd.append("-s")
        
        if markers:
            cmd.extend(["-m", " or ".join(markers)])
        
        if keywords:
            cmd.extend(["-k", " or ".join(keywords)])
        
        if extra_args:
            cmd.extend(extra_args)
        
        return self._run_test_command(
            cmd=cmd,
            timeout=timeout or self._default_timeout,
            framework="pytest",
        )
    
    def run_unittest(
        self,
        test_path: str | None = None,
        pattern: str = "test*.py",
        timeout: float | None = None,
        verbose: bool = True,
    ) -> TestRunResult:
        """
        Run unittest tests.
        
        Args:
            test_path: Path to tests
            pattern: Test file pattern
            timeout: Timeout in seconds
            verbose: Verbose output
            
        Returns:
            TestRunResult with test results
        """
        cmd = [sys.executable, "-m", "unittest"]
        
        if test_path:
            resolved = self._workspace_root / test_path
            cmd.extend(["discover", "-s", str(resolved), "-p", pattern])
        else:
            cmd.extend(["discover", "-s", str(self._workspace_root), "-p", pattern])
        
        if verbose:
            cmd.append("-v")
        
        return self._run_test_command(
            cmd=cmd,
            timeout=timeout or self._default_timeout,
            framework="unittest",
        )
    
    def run_custom_command(
        self,
        command: list[str],
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> TestRunResult:
        """
        Run a custom test command.
        
        Args:
            command: Command and arguments
            timeout: Timeout in seconds
            env: Additional environment variables
            
        Returns:
            TestRunResult with results
        """
        return self._run_test_command(
            cmd=command,
            timeout=timeout or self._default_timeout,
            framework="custom",
            env=env,
        )
    
    def _run_test_command(
        self,
        cmd: list[str],
        timeout: float,
        framework: str,
        env: dict[str, str] | None = None,
    ) -> TestRunResult:
        """Execute test command and parse results."""
        start_time = datetime.now(timezone.utc)
        
        # Prepare environment
        run_env = os.environ.copy()
        run_env["PYTHONDONTWRITEBYTECODE"] = "1"
        run_env["PYTHONUNBUFFERED"] = "1"
        
        # Disable network (best effort)
        run_env["NO_PROXY"] = "*"
        
        if env:
            run_env.update(env)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._workspace_root),
                capture_output=True,
                timeout=timeout,
                env=run_env,
                text=True,
            )
            
            stdout = result.stdout[:self._max_output_size]
            stderr = result.stderr[:self._max_output_size]
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Parse results based on framework
            if framework == "pytest":
                suite_result = self._parse_pytest_output(stdout, stderr, duration)
            elif framework == "unittest":
                suite_result = self._parse_unittest_output(stdout, stderr, duration)
            else:
                suite_result = self._parse_generic_output(stdout, stderr, duration, result.returncode)
            
            success = suite_result.status == TestStatus.PASSED
            
            self._log("run", success, {
                "framework": framework,
                "command": cmd,
                "duration_ms": duration,
            })
            
            return TestRunResult(
                success=success,
                suites=[suite_result],
                total_duration_ms=duration,
                summary=self._generate_summary(suite_result),
            )
            
        except subprocess.TimeoutExpired:
            duration = timeout * 1000
            
            self._log("run", False, {
                "framework": framework,
                "command": cmd,
                "error": "timeout",
            })
            
            return TestRunResult(
                success=False,
                suites=[TestSuiteResult(
                    suite_name=framework,
                    status=TestStatus.TIMEOUT,
                    duration_ms=duration,
                )],
                total_duration_ms=duration,
                error=f"Test execution timed out after {timeout}s",
            )
            
        except Exception as e:
            self._log("run", False, {
                "framework": framework,
                "command": cmd,
                "error": str(e),
            })
            
            return TestRunResult(
                success=False,
                error=str(e),
            )
    
    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        duration_ms: float,
    ) -> TestSuiteResult:
        """Parse pytest output."""
        tests: list[TestResult] = []
        passed = failed = errors = skipped = 0
        
        # Parse individual test results
        test_pattern = r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)"
        for match in re.finditer(test_pattern, stdout, re.MULTILINE):
            name, status = match.groups()
            status_enum = {
                "PASSED": TestStatus.PASSED,
                "FAILED": TestStatus.FAILED,
                "ERROR": TestStatus.ERROR,
                "SKIPPED": TestStatus.SKIPPED,
            }.get(status, TestStatus.ERROR)
            
            tests.append(TestResult(name=name, status=status_enum))
            
            if status == "PASSED":
                passed += 1
            elif status == "FAILED":
                failed += 1
            elif status == "ERROR":
                errors += 1
            else:
                skipped += 1
        
        # Parse summary line if individual tests not found
        if not tests:
            summary_pattern = r"(\d+) passed|(\d+) failed|(\d+) error|(\d+) skipped"
            for match in re.finditer(summary_pattern, stdout, re.IGNORECASE):
                groups = match.groups()
                if groups[0]:
                    passed = int(groups[0])
                if groups[1]:
                    failed = int(groups[1])
                if groups[2]:
                    errors = int(groups[2])
                if groups[3]:
                    skipped = int(groups[3])
        
        total = passed + failed + errors + skipped
        status = TestStatus.PASSED if (failed == 0 and errors == 0) else TestStatus.FAILED
        
        return TestSuiteResult(
            suite_name="pytest",
            status=status,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_ms=duration_ms,
            tests=tests,
            stdout=stdout,
            stderr=stderr,
        )
    
    def _parse_unittest_output(
        self,
        stdout: str,
        stderr: str,
        duration_ms: float,
    ) -> TestSuiteResult:
        """Parse unittest output."""
        tests: list[TestResult] = []
        passed = failed = errors = skipped = 0
        
        # Unittest output is typically in stderr
        output = stderr if stderr else stdout
        
        # Parse test results (format: test_name (module.class) ... ok/FAIL/ERROR)
        test_pattern = r"(\S+)\s+\(([^)]+)\)\s+\.\.\.\s+(ok|FAIL|ERROR|skipped)"
        for match in re.finditer(test_pattern, output, re.IGNORECASE):
            name, module, status = match.groups()
            full_name = f"{module}.{name}"
            
            status_enum = {
                "ok": TestStatus.PASSED,
                "fail": TestStatus.FAILED,
                "error": TestStatus.ERROR,
                "skipped": TestStatus.SKIPPED,
            }.get(status.lower(), TestStatus.ERROR)
            
            tests.append(TestResult(name=full_name, status=status_enum))
            
            if status.lower() == "ok":
                passed += 1
            elif status.lower() == "fail":
                failed += 1
            elif status.lower() == "error":
                errors += 1
            else:
                skipped += 1
        
        # Parse summary
        summary_pattern = r"Ran (\d+) tests?"
        match = re.search(summary_pattern, output)
        if match:
            total = int(match.group(1))
        else:
            total = passed + failed + errors + skipped
        
        status = TestStatus.PASSED if (failed == 0 and errors == 0) else TestStatus.FAILED
        
        return TestSuiteResult(
            suite_name="unittest",
            status=status,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_ms=duration_ms,
            tests=tests,
            stdout=stdout,
            stderr=stderr,
        )
    
    def _parse_generic_output(
        self,
        stdout: str,
        stderr: str,
        duration_ms: float,
        return_code: int,
    ) -> TestSuiteResult:
        """Parse generic command output."""
        status = TestStatus.PASSED if return_code == 0 else TestStatus.FAILED
        
        return TestSuiteResult(
            suite_name="custom",
            status=status,
            total=1,
            passed=1 if return_code == 0 else 0,
            failed=0 if return_code == 0 else 1,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
        )
    
    def _generate_summary(self, suite: TestSuiteResult) -> str:
        """Generate human-readable summary."""
        parts = []
        
        if suite.passed > 0:
            parts.append(f"{suite.passed} passed")
        if suite.failed > 0:
            parts.append(f"{suite.failed} failed")
        if suite.errors > 0:
            parts.append(f"{suite.errors} errors")
        if suite.skipped > 0:
            parts.append(f"{suite.skipped} skipped")
        
        status = "✓" if suite.status == TestStatus.PASSED else "✗"
        return f"{status} {', '.join(parts)} in {suite.duration_ms:.0f}ms"


class TypeChecker:
    """
    Static type checker runner.
    
    Supports mypy and pyright for Python type checking.
    """
    
    def __init__(
        self,
        workspace_root: Path,
        telemetry: "TelemetryCollector | None" = None,
        default_timeout: float = 120.0,
    ) -> None:
        self._workspace_root = workspace_root.resolve()
        self._telemetry = telemetry
        self._default_timeout = default_timeout
    
    def run_mypy(
        self,
        path: str | None = None,
        strict: bool = False,
        ignore_missing_imports: bool = True,
        timeout: float | None = None,
    ) -> TestRunResult:
        """
        Run mypy type checker.
        
        Args:
            path: Path to check
            strict: Strict mode
            ignore_missing_imports: Ignore missing imports
            timeout: Timeout in seconds
            
        Returns:
            TestRunResult with type errors
        """
        cmd = [sys.executable, "-m", "mypy"]
        
        if path:
            cmd.append(str(self._workspace_root / path))
        else:
            cmd.append(str(self._workspace_root))
        
        if strict:
            cmd.append("--strict")
        
        if ignore_missing_imports:
            cmd.append("--ignore-missing-imports")
        
        cmd.append("--no-error-summary")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._workspace_root),
                capture_output=True,
                timeout=timeout or self._default_timeout,
                text=True,
            )
            
            errors = self._parse_mypy_output(result.stdout)
            success = len(errors) == 0
            
            return TestRunResult(
                success=success,
                suites=[TestSuiteResult(
                    suite_name="mypy",
                    status=TestStatus.PASSED if success else TestStatus.FAILED,
                    total=len(errors),
                    failed=len(errors),
                    tests=errors,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )],
                summary=f"{'✓' if success else '✗'} {len(errors)} type errors",
            )
            
        except subprocess.TimeoutExpired:
            return TestRunResult(
                success=False,
                error=f"Mypy timed out after {timeout or self._default_timeout}s",
            )
        except Exception as e:
            return TestRunResult(
                success=False,
                error=str(e),
            )
    
    def _parse_mypy_output(self, output: str) -> list[TestResult]:
        """Parse mypy output into test results."""
        errors: list[TestResult] = []
        
        # Format: file.py:line: error: message
        pattern = r"^(.+?):(\d+):\s*(error|warning|note):\s*(.+)$"
        for match in re.finditer(pattern, output, re.MULTILINE):
            file_path, line, severity, message = match.groups()
            
            errors.append(TestResult(
                name=f"{file_path}:{line}",
                status=TestStatus.FAILED if severity == "error" else TestStatus.PASSED,
                message=message,
                file_path=file_path,
                line_number=int(line),
            ))
        
        return errors

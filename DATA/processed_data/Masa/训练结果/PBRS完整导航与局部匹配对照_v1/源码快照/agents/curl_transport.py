"""Windows Schannel curl备用传输：保留证书验证，凭据/图像通过stdin，不进argv或磁盘。"""
import json
import shutil
import subprocess

import httpx


class CurlTransport:
    def __init__(self, timeout=90):
        self.timeout = timeout
        self.executable = shutil.which("curl")
        if self.executable is None:
            raise ValueError("系统curl不可用")

    @staticmethod
    def quoted(value):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('\r', '\\r').replace('\n', '\\n') + '"'

    def request(self, method, url, headers, json=None):
        if any('\r' in value or '\n' in value for value in headers.values()):
            raise ValueError("非法header换行")
        lines = ["silent", "show-error", "proto = \"=https\"",
                 "url = " + self.quoted(url), "request = " + self.quoted(method),
                 "connect-timeout = 15", f"max-time = {self.timeout}",
                 'write-out = "\\n__STATUS__%{http_code}"']
        for name, value in headers.items():
            lines.append("header = " + self.quoted(name + ": " + value))
        if json is not None:
            lines.append('header = "Content-Type: application/json"')
            lines.append("data-binary = " + self.quoted(__import__('json').dumps(json)))
        try:
            process = subprocess.run([self.executable, "--disable", "--config", "-"],
                                     input=('\n'.join(lines)+'\n').encode('utf-8'),
                                     capture_output=True, timeout=self.timeout+5)
        except subprocess.TimeoutExpired:
            raise httpx.ReadTimeout("curl timeout") from None
        if process.returncode:
            error = httpx.ReadTimeout if process.returncode == 28 else httpx.ConnectError
            raise error(f"curl transport exit {process.returncode}")
        try:
            content, code = process.stdout.rsplit(b"\n__STATUS__", 1)
            status = int(code)
        except (ValueError, IndexError):
            raise httpx.ProtocolError("curl missing status marker") from None
        return httpx.Response(status, content=content)

    def close(self):
        pass

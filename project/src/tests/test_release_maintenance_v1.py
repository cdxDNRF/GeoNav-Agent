"""PLATFORM-029: real failure branches with synthetic transports; no models."""
from contextlib import nullcontext
import errno
from io import BytesIO
import json
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request
from project.src.tools import release_maintenance_v1 as maintenance
from project.src.tests.test_spatial_evidence_v1 import Tests as OriginalEvidenceTests


class PortTests(unittest.TestCase):
    def test_explicit_refusal_is_free_and_does_not_attempt_http(self):
        with patch.object(maintenance.socket,'create_connection',side_effect=ConnectionRefusedError(errno.ECONNREFUSED,'refused')), patch.object(maintenance.urllib.request,'build_opener') as http:
            self.assertEqual(maintenance.require_free(8769)['state'],'free')
            http.assert_not_called()

    def test_timeout_denied_and_unknown_os_error_fail_closed(self):
        for exc in [TimeoutError(),PermissionError(errno.EACCES,'denied'),OSError(errno.EHOSTUNREACH,'unreachable')]:
            with self.subTest(exc=type(exc).__name__), patch.object(maintenance.socket,'create_connection',side_effect=exc),patch.object(maintenance.socket,'socket',side_effect=PermissionError(errno.EACCES,'cannot prove exclusive bind')):
                self.assertEqual(maintenance.probe_port(8769)['state'],'unknown')
                with self.assertRaises(RuntimeError):maintenance.require_free(8769)

    def test_filtered_closed_port_needs_successful_exclusive_os_bind(self):
        sock=Mock();sock.__enter__=Mock(return_value=sock);sock.__exit__=Mock(return_value=False)
        with patch.object(maintenance.socket,'create_connection',side_effect=TimeoutError()),patch.object(maintenance.socket,'socket',return_value=sock):
            self.assertEqual(maintenance.require_free(8769)['reason'],'exclusive_bind_succeeded_after_connect_timeout')
            sock.bind.assert_called_once_with(('127.0.0.1',8769))

    def test_listening_http_failure_never_means_free(self):
        for exc in [urllib.error.HTTPError('http://127.0.0.1',503,'busy',{},None),TimeoutError(),urllib.error.URLError(PermissionError(13,'denied'))]:
            op=Mock();op.open.side_effect=exc
            with self.subTest(exc=type(exc).__name__), patch.object(maintenance.socket,'create_connection',return_value=nullcontext()), patch.object(maintenance.urllib.request,'build_opener',return_value=op):
                self.assertEqual(maintenance.probe_port(8769)['state'],'occupied')
                with self.assertRaises(RuntimeError):maintenance.require_free(8769)

    def test_current_service_is_reused_but_never_free_for_release(self):
        def response(*args,**kwargs):return BytesIO(json.dumps({'version':maintenance.VERSION}).encode())
        op=Mock();op.open.side_effect=response
        with patch.object(maintenance.socket,'create_connection',return_value=nullcontext()),patch.object(maintenance.urllib.request,'build_opener',return_value=op):
            self.assertEqual(maintenance.probe_port(8769)['state'],'current')
            with self.assertRaises(RuntimeError):maintenance.require_free(8769)

    def test_listening_non_json_or_other_version_remains_occupied(self):
        for body in [b'not JSON',b'[]',b'{"version":"other"}']:
            op=Mock();op.open.side_effect=lambda *a,**kw:BytesIO(body)
            with self.subTest(body=body),patch.object(maintenance.socket,'create_connection',return_value=nullcontext()),patch.object(maintenance.urllib.request,'build_opener',return_value=op):
                self.assertEqual(maintenance.probe_port(8769)['state'],'occupied')


class EvidenceTests(OriginalEvidenceTests):
    def test_http_readonly_tokens_existing_pages_and_same_origin(self):
        from project.src.platform_core.service_v11 import Workbench
        from project.src.webapp.workbench_server_v21 import make_server
        op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with tempfile.TemporaryDirectory(prefix='http-',dir=maintenance.OUT/'测试') as store:
            w=Workbench(maintenance.ROOT,store,bindings=False)
            server=make_server(w,0);t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                for page in ['/','/scenes','/ground-live','/ground-walk','/perception','/spatial-evidence']:
                    with op.open(url+page) as f:self.assertIn(b'/spatial-evidence',f.read())
                with op.open(url+'/api/meta') as f:
                    m=json.load(f);self.assertEqual(m['version'],maintenance.VERSION);self.assertTrue(m['scene_token']);self.assertTrue(m['workflow_token'])
                with op.open(url+'/spatial-evidence/app.js') as f:
                    script=f.read().decode();self.assertIn("[['original-image','original']]",script)
                with op.open(url+'/api/spatial-evidence/export?case=13&method=Spatial&seed=0&arm=Wrong') as f:
                    self.assertEqual(json.load(f)['record']['new_model_calls'],0)
                self.assertEqual(maintenance.probe_port(server.server_port)['state'],'current')
                for req,code in [(urllib.request.Request(url+'/api/spatial-evidence/catalog',headers={'Origin':'https://untrusted.invalid'}),403),
                                 (urllib.request.Request(url+'/api/spatial-evidence/detail',data=b'{}',headers={'Content-Type':'application/json'}),405)]:
                    with self.assertRaises(urllib.error.HTTPError) as e:op.open(req)
                    self.assertEqual(e.exception.code,code)
            finally:server.shutdown();server.server_close();w.shutdown();t.join(3)


del OriginalEvidenceTests
if __name__=='__main__':unittest.main()

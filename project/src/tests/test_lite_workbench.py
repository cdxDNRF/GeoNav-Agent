"""Automated end-to-end integration tests for GeoNav Lite workbench."""
import base64
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'project/src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'project/src'))

from PIL import Image, ImageDraw
from webapp.lite_server import (
    LiteServerManager,
    _explain_cloud_exception,
    make_lite_server,
    parse_decision_enhanced,
)


class TestLiteWorkbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = LiteServerManager(ROOT)
        cls.port = 8792  # 专用测试端口
        cls.server = make_lite_server(cls.manager, cls.port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(1.0)
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        try:
            cls.server.shutdown()
            cls.server.server_close()
        except Exception:
            pass
        try:
            cls.manager.shutdown()
        except Exception:
            pass

    def _get(self, path):
        req = Request(f"{self.base_url}{path}")
        with urlopen(req, timeout=30) as res:
            return res.status, res.headers.get_content_type(), res.read()

    def _post_json(self, path, payload):
        data = json.dumps(payload).encode('utf-8')
        req = Request(f"{self.base_url}{path}", data=data, headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(req, timeout=90) as res:
                return res.status, res.headers.get_content_type(), json.loads(res.read().decode('utf-8'))
        except HTTPError as e:
            err_body = e.read().decode('utf-8')
            raise RuntimeError(f"HTTPError {e.code}: {err_body}") from e

    def test_01_static_assets_and_clean_ui(self):
        """测试 HTML/CSS/JS 静态资源、新增控件与去 emoji 清洁度"""
        assets = {}
        for route, exp_mime in [
            ('/', 'text/html'),
            ('/geonav_lite.css', 'text/css'),
            ('/geonav_lite.js', 'text/javascript')
        ]:
            st, mime, body = self._get(route)
            self.assertEqual(st, 200)
            self.assertEqual(mime, exp_mime)
            text = body.decode('utf-8')
            self.assertGreater(len(text), 200)
            assets[route] = text

        html = assets['/']
        self.assertIn('id="btn-toggle-eye"', html)
        self.assertIn('id="upload-grid-selector"', html)
        self.assertIn('id="toast-container"', html)
        self.assertIn('id="btn-random-aerial"', html)
        self.assertIn('id="btn-test-api"', html)
        for noisy in ('🧭', '📚', '🏁', '🎯', '🧠', '🤖', '🗺️', '🎉'):
            self.assertNotIn(noisy, html)
            self.assertNotIn(noisy, assets['/geonav_lite.css'])
            self.assertNotIn(noisy, assets['/geonav_lite.js'])

    def test_02_curated_multi_image_aerial_catalog(self):
        """测试多来源航拍图像大幅扩容且严格保证单图至多5道合格题"""
        st, _, body = self._get('/api/meta')
        self.assertEqual(st, 200)
        data = json.loads(body.decode('utf-8'))
        tasks = data['tasks']

        aerial_tasks = [t for t in tasks if t.get('dataset_id') == 'aerial_bank']
        cam_tasks = [t for t in tasks if t.get('dataset_id') == 'camera-development']
        self.assertEqual(len(cam_tasks), 192)

        distinct_imgs = {(t['bank_id'], t['area']) for t in aerial_tasks}
        self.assertGreaterEqual(len(distinct_imgs), 190, "至少包含 190 张独立航拍图片")
        self.assertGreaterEqual(len(aerial_tasks), 900)

        # 严厉核验：每张独立底图最多只有 <= 5 道预置题目
        per_img_counts = Counter((t['bank_id'], t['area'], t['grid']) for t in aerial_tasks)
        self.assertLessEqual(max(per_img_counts.values()), 5)

        banks_seen = {t['bank_id'] for t in aerial_tasks}
        for b in ('masa-s2', 'masa-s3', 'swissview-1', 'swissview-2', 'massgis', 'masa-train'):
            self.assertIn(b, banks_seen)

        grid_ks = [g['k'] for g in data.get('upload_grids', [])]
        for k in (3, 4, 5, 6, 7, 10):
            self.assertIn(k, grid_ks)

    def test_03_multi_bank_m0_and_observer_eye_isolation(self):
        """测试 Masa S3、SwissView 瑞士图与 MassGIS 10×10 执行及小眼睛全图隔离"""
        _, _, meta = self._get('/api/meta')
        tasks = json.loads(meta.decode('utf-8'))['tasks']

        # 1) 测试 SwissView 5x5 + 小眼睛隔离
        sv_task = next(t for t in tasks if t.get('bank_id') == 'swissview-1')
        st, _, res = self._post_json('/api/create', {
            'task_key': sv_task['task_key'],
            'policy': 'M0',
            'start': 0,
            'goal': 24
        })
        self.assertEqual(st, 200)
        self.assertEqual(res['visited'], [0])

        obs_st, obs_mime, obs_bytes = self._get(
            f"/api/image?task_key={sv_task['task_key']}&cell=24&observer=1"
        )
        self.assertEqual(obs_st, 200)
        self.assertEqual(obs_mime, 'image/png')
        self.assertGreater(len(obs_bytes), 200)

        _, _, state_raw = self._get('/api/state')
        self.assertEqual(json.loads(state_raw.decode('utf-8'))['visited'], [0])

        st, _, step_res = self._post_json('/api/step', {})
        self.assertEqual(step_res['step'], 1)
        self.assertEqual(len(step_res['decision']['probabilities']), 4)

        # 2) 测试 MassGIS 10x10 大区域 M0 + 径向前瞻
        mg10_task = next(t for t in tasks if t.get('bank_id') == 'massgis' and t.get('grid') == 10)
        st, _, res10 = self._post_json('/api/create', {
            'task_key': mg10_task['task_key'],
            'policy': 'M0'
        })
        self.assertEqual(res10['grid'], 10)
        self.assertEqual(res10['budget'], 20)
        st, _, step10 = self._post_json('/api/step', {})
        self.assertEqual(step10['step'], 1)

        # 3) 测试 Masa Train 精选集单步与 Reset 清零
        tr_task = next(t for t in tasks if t.get('bank_id') == 'masa-train')
        self._post_json('/api/create', {'task_key': tr_task['task_key'], 'policy': 'M0'})
        st, _, tr_step = self._post_json('/api/step', {})
        self.assertEqual(tr_step['step'], 1)

        st, _, rst = self._post_json('/api/reset', {})
        self.assertTrue(rst['reset'])
        _, _, empty_raw = self._get('/api/state')
        self.assertEqual(json.loads(empty_raw.decode('utf-8')), {})

    def test_04_multi_grid_upload_m0_and_volatility(self):
        """测试多规格 n*n 自定义图片纯内存切片、M0 与 Frontier 推演及临时清除机制"""
        im = Image.new('RGB', (720, 720), color=(90, 140, 180))
        draw = ImageDraw.Draw(im)
        draw.rectangle([100, 100, 340, 340], fill=(210, 180, 90))
        draw.line([0, 0, 720, 720], fill=(240, 240, 240), width=12)
        buf = BytesIO()
        im.save(buf, format='PNG')
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

        st, _, up3 = self._post_json('/api/upload', {
            'image': b64,
            'name': '临时3x3测试图',
            'grid': 3
        })
        self.assertEqual(st, 200)
        self.assertEqual(up3['grid'], 3)
        self.assertEqual(up3['budget'], 6)

        st, _, s3 = self._post_json('/api/create', {
            'task_key': up3['task_key'],
            'policy': 'M0',
            'start': 0,
            'goal': 8
        })
        self.assertEqual(s3['grid'], 3)
        st, _, step3 = self._post_json('/api/step', {})
        self.assertEqual(step3['step'], 1)
        self.assertEqual(len(step3['decision']['probabilities']), 4)

        prev_st, prev_mime, prev_bytes = self._get(
            f"/api/image?task_key={up3['task_key']}&observer=1&full=1"
        )
        self.assertEqual(prev_st, 200)
        self.assertEqual(prev_mime, 'image/png')
        self.assertGreater(len(prev_bytes), 500)

        st, _, up10 = self._post_json('/api/upload', {
            'image': b64,
            'name': '临时10x10大区域',
            'grid': 10
        })
        self.assertEqual(up10['grid'], 10)
        self.assertEqual(up10['budget'], 20)
        self._post_json('/api/create', {'task_key': up10['task_key'], 'policy': 'frontier', 'start': 0, 'goal': 99})
        st, _, step10 = self._post_json('/api/step', {})
        self.assertEqual(step10['step'], 1)

        orig_has_clip = self.manager._has_clip_encoder
        self.manager._has_clip_encoder = lambda: False
        try:
            self._post_json('/api/create', {'task_key': up3['task_key'], 'policy': 'M0', 'start': 0, 'goal': 8})
            st, _, step_noclip = self._post_json('/api/step', {})
            self.assertEqual(step_noclip['step'], 1)
        finally:
            self.manager._has_clip_encoder = orig_has_clip

        self._post_json('/api/upload/delete', {'task_key': up3['task_key']})
        self._post_json('/api/upload/delete', {'task_key': up10['task_key']})

        fresh_mgr = LiteServerManager(ROOT)
        try:
            self.assertEqual(len(fresh_mgr.uploads), 0)
            self.assertFalse(any(t.get('dataset_id') == 'upload' for t in fresh_mgr.tasks_catalog))
        finally:
            fresh_mgr.shutdown()

    def test_05_camera_session_and_image_endpoint(self):
        """测试地面 OpenCVL 任务创建、单步、目标照片与俯视全图预览渲染"""
        _, _, meta = self._get('/api/meta')
        tasks = json.loads(meta.decode('utf-8'))['tasks']
        first_cam = next(t for t in tasks if t['dataset_id'] == 'camera-development')

        st, _, cam_res = self._post_json('/api/create', {
            'task_key': first_cam['task_key'],
            'policy': 'frontier'
        })
        self.assertEqual(st, 200)
        self.assertEqual(cam_res['type'], 'camera')

        st, _, cam_step = self._post_json('/api/step', {})
        self.assertEqual(cam_step['step'], 1)

        t_st, t_mime, t_bytes = self._get('/api/image?target=1')
        self.assertEqual(t_st, 200)
        self.assertEqual(t_mime, 'image/png')
        self.assertGreater(len(t_bytes), 200)

        c_st, _, c_bytes = self._get(f"/api/image?cell={cam_step['position']}")
        self.assertEqual(c_st, 200)
        self.assertGreater(len(c_bytes), 200)

    def test_06_api_config_and_diagnostics(self):
        """测试云端 API 配置保存、密钥脱敏与精确错误诊断解析"""
        st, _, cfg = self._post_json('/api/config', {
            'base_url': 'https://api.deepseek.com/v1',
            'model': 'deepseek-flash',
            'api_key': 'sk-testmockkeyforvalidationonly1234567890',
            'timeout': 25,
            'disable_thinking': True
        })
        self.assertEqual(st, 200)
        self.assertEqual(cfg['model'], 'deepseek-flash')
        self.assertEqual(cfg['max_tokens'], 2048)
        self.assertTrue(cfg['disable_thinking'])
        self.assertNotIn('sk-testmockkeyforvalidationonly1234567890', json.dumps(cfg))

        from webapp.cloud_v3 import CloudConfig
        from webapp.lite_server import _prepare_cloud_payload, _execute_cloud_with_adaptation
        ds_cfg = CloudConfig('https://api.deepseek.com/v1', 'deepseek-flash', 'sk-dummy', 30.0, 2048)
        adapted = _prepare_cloud_payload({'model': 'deepseek-flash', 'max_tokens': 512}, ds_cfg, disable_thinking=True)
        self.assertEqual(adapted.get('thinking'), {'type': 'disabled'})
        self.assertGreaterEqual(adapted['max_tokens'], 2048)

        oa_cfg = CloudConfig('https://api.openai.com/v1', 'gpt-4o-mini', 'sk-dummy', 30.0, 2048)
        adapted_oa = _prepare_cloud_payload({'model': 'gpt-4o-mini', 'max_tokens': 512}, oa_cfg, disable_thinking=True)
        self.assertNotIn('thinking', adapted_oa)
        self.assertGreaterEqual(adapted_oa['max_tokens'], 4096)

        # 验证截断自动重试提额与 usage 诊断显示
        seen_budgets = []
        import webapp.lite_server as ls_mod
        orig_req = ls_mod.lite_cloud_request
        try:
            def fake_req(_cfg, pl):
                seen_budgets.append(pl['max_tokens'])
                if pl['max_tokens'] < 8192:
                    return {'choices': [{'finish_reason': 'length', 'message': {'content': ''}}],
                            'usage': {'completion_tokens': pl['max_tokens'], 'completion_tokens_details': {'reasoning_tokens': pl['max_tokens']}}}, None
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"action":"up"}'}}],
                        'usage': {'completion_tokens': 24}}, None
            ls_mod.lite_cloud_request = fake_req
            from webapp.cloud_v3 import parse as cparse
            _, _, parsed = _execute_cloud_with_adaptation(ds_cfg, {'model': 'deepseek-flash', 'max_tokens': 1024}, disable_thinking=True, parser=lambda r: cparse(r, ['up', 'down']))
            self.assertEqual(parsed['action'], 'up')
            self.assertEqual(seen_budgets, [2048, 8192])
        finally:
            ls_mod.lite_cloud_request = orig_req

        # 验证 HTTP 402 等常见云端状态能给出明确汉字原因而非笼统文案
        msg_402 = _explain_cloud_exception(RuntimeError('err'), http_status=402)
        self.assertIn('402', msg_402)
        dec_fb = parse_decision_enhanced({
            'cloud': {'model': 'gemma4:31b', 'fallback': True, 'detailed_error': msg_402}
        }, 'down', actor='agent', policy='Cloud')
        self.assertIn('402', dec_fb['reason_detail'])
        self.assertIn('云端失败回退本地建议', dec_fb['reason_headline'])


if __name__ == '__main__':
    unittest.main()

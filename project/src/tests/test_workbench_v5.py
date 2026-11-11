"""End-to-end regression tests for GeoNav V5.0 Unified SPA Workbench (service_v17 + workbench_server_v28)."""
import base64
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw
from project.src.platform_core.service_v17 import Workbench, BENCHMARK_OVERVIEW
from project.src.webapp.workbench_server_v28 import make_server, VERSION

ROOT = Path(__file__).resolve().parents[3]
TEMP_DIR = ROOT / 'temp'
OWNER = 'a1b2c3d4e5f60718293a4b5c6d7e8f90'


class WorkbenchV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=TEMP_DIR)
        cls.store = Path(cls.tmp.name).resolve()
        assert cls.store.is_relative_to(TEMP_DIR.resolve())
        cls.workbench = Workbench(ROOT, cls.store)
        cls.server = make_server(cls.workbench, port=0)
        cls.port = cls.server.server_port
        cls.base_url = f'http://127.0.0.1:{cls.port}'
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.workbench.shutdown()
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def _get(self, path, headers=None):
        req = Request(self.base_url + path, headers=headers or {'X-GeoNav-Session': OWNER})
        with urlopen(req, timeout=30) as resp:
            return resp.status, resp.headers.get('Content-Type', ''), resp.read()

    def _post(self, path, payload, token):
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'X-GeoNav-Session': OWNER,
            'X-GeoNav-Scene-Token': token,
        }
        req = Request(self.base_url + path, data=data, headers=headers, method='POST')
        with urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))

    def test_01_spa_routes_and_static_assets(self):
        for route in ('/', '/v5', '/controllers'):
            status, mime, body = self._get(route)
            self.assertEqual(status, 200)
            self.assertIn('text/html', mime)
            text = body.decode('utf-8')
            self.assertIn('GeoNav V5.0', text)
            for ws_id in ('ws-overview', 'ws-navigator', 'ws-sandbox', 'ws-arena', 'ws-ground', 'ws-evidence'):
                self.assertIn(f'id="{ws_id}"', text)
            self.assertIn('id="wizard-bar"', text)

        status, mime, css_body = self._get('/v5/style.css')
        self.assertEqual(status, 200)
        self.assertIn('text/css', mime)
        self.assertIn('--v5-primary', css_body.decode('utf-8'))

        status, mime, js_body = self._get('/v5/app.js')
        self.assertEqual(status, 200)
        self.assertIn('text/javascript', mime)
        self.assertIn('WIZARD_STEPS', js_body.decode('utf-8'))

        # Legacy rollback entry remains intact
        status, mime, legacy_body = self._get('/legacy/controllers')
        self.assertEqual(status, 200)
        self.assertIn('text/html', mime)
        self.assertIn('batch-table', legacy_body.decode('utf-8'))

    def test_02_meta_and_v5_overview(self):
        status, _, body = self._get('/api/meta')
        self.assertEqual(status, 200)
        meta = json.loads(body.decode('utf-8'))
        self.assertEqual(meta['version'], VERSION)
        self.assertTrue(meta.get('scene_token'))
        self.assertEqual(meta['capabilities']['kernel'], 'geonav-core-v17')
        self.assertTrue(meta['capabilities']['unified_v5_spa'])

        status, _, ov_body = self._get('/api/v5/overview')
        self.assertEqual(status, 200)
        ov = json.loads(ov_body.decode('utf-8'))
        self.assertEqual(ov['version'], 'geonav-workbench-v5.0')
        self.assertEqual(len(ov['benchmark_overview']), len(BENCHMARK_OVERVIEW))
        self.assertGreaterEqual(len(ov['assets']), 5)
        self.assertGreaterEqual(ov['counts']['aerial_tasks'], 10)
        self.assertGreaterEqual(ov['counts']['capturepoint_tasks'], 192)

    def test_03_all_six_workspace_catalogs_and_execution(self):
        _, _, meta_raw = self._get('/api/meta')
        token = json.loads(meta_raw.decode('utf-8'))['scene_token']

        # 1. Navigator (controllers)
        _, _, c_raw = self._get(f'/api/controllers/catalog?owner={OWNER}')
        c_cat = json.loads(c_raw.decode('utf-8'))
        self.assertGreaterEqual(len(c_cat['datasets']), 2)
        masa_tasks = [t for t in c_cat['tasks'] if t['dataset_id'] == 'masa-s2']
        self.assertEqual(len(masa_tasks), 100)

        status, created = self._post('/api/controllers/create', {
            'name': 'v5_test_run',
            'dataset_id': 'masa-s2',
            'task_key': masa_tasks[0]['task_key'],
            'controller': 'M0',
            'seed': 0,
        }, token)
        self.assertEqual(status, 200)
        self.assertEqual(created['step'], 0)

        status, stepped = self._post('/api/controllers/step', {
            'id': created['id'],
            'kind': created['kind'],
            'action': None,
        }, token)
        self.assertEqual(status, 200)
        self.assertEqual(stepped['step'], 1)
        self._post('/api/controllers/release', {}, token)

        # 2. Sandbox (scenes)
        _, _, s_raw = self._get(f'/api/scenes/catalog?owner={OWNER}')
        s_cat = json.loads(s_raw.decode('utf-8'))
        self.assertGreaterEqual(len(s_cat['datasets']), 1)
        ds0 = s_cat['datasets'][0]
        t0 = ds0['tasks'][0]
        status, scene = self._post('/api/scenes/create', {
            'name': 'v5_sandbox_test',
            'dataset_id': ds0['id'],
            'base_task_key': t0['key'],
            'start': t0['start'],
            'goal': t0['goal'],
            'budget': ds0['budget'],
            'seed': 0,
        }, token)
        self.assertEqual(status, 200)
        status, run = self._post('/api/scenes/start', {'scene_id': scene['id']}, token)
        self.assertEqual(status, 200)
        status, run = self._post('/api/scenes/step', {'id': run['id'], 'action': 'right'}, token)
        self.assertEqual(status, 200)
        self.assertEqual(len(run['events']), 1)
        self._post('/api/scenes/release', {}, token)

        # 3. Arena (api-comparisons & controller-batches)
        status, _, api_raw = self._get(f'/api/api-comparisons/catalog?owner={OWNER}')
        self.assertEqual(status, 200)
        self.assertIn('comparisons', json.loads(api_raw.decode('utf-8')))

        status, _, b_raw = self._get(f'/api/controller-batches/catalog?owner={OWNER}')
        self.assertEqual(status, 200)
        self.assertIn('batches', json.loads(b_raw.decode('utf-8')))

        # 4. Ground (ground-walk & ground-live)
        status, _, gw_raw = self._get(f'/api/ground-walk/catalog?owner={OWNER}')
        self.assertEqual(status, 200)
        self.assertIn('saved', json.loads(gw_raw.decode('utf-8')))

        status, _, gl_raw = self._get(f'/api/ground-live/catalog?owner={OWNER}')
        self.assertEqual(status, 200)
        self.assertIn('saved', json.loads(gl_raw.decode('utf-8')))

        # 5. Evidence (perception & spatial-evidence)
        status, _, p_raw = self._get('/api/perception/catalog')
        self.assertEqual(status, 200)
        p_cat = json.loads(p_raw.decode('utf-8'))
        self.assertEqual(len(p_cat['cases']), 20)

        status, _, se_raw = self._get('/api/spatial-evidence/catalog')
        self.assertEqual(status, 200)
        se_cat = json.loads(se_raw.decode('utf-8'))
        self.assertEqual(len(se_cat['cases']), 150)

    def test_04_emoji_free_and_popover_ui(self):
        _, _, html_raw = self._get('/')
        _, _, css_raw = self._get('/v5/style.css')
        _, _, js_raw = self._get('/v5/app.js')
        html = html_raw.decode('utf-8')
        css = css_raw.decode('utf-8')
        js = js_raw.decode('utf-8')

        for noisy in (
            '🏠', '🚁', '🗺️', '⚔️', '📷', '🔬', '🛡️', '🚀', '🎯', '⏱',
            '📊', '🧩', '⚡', '🌙', '☀️', '✨', '📂', '💾', '📌', '🔀',
            '✅', '❌', '⚠️', '👤', '🤖', '👁️', '🛰️', '🎉', '📥', '⚖️',
            '🚶', '🔍', '🧭',
        ):
            self.assertNotIn(noisy, html)
            self.assertNotIn(noisy, css)
            self.assertNotIn(noisy, js)

        for marker in ('v5-tip-btn', 'id="v5-popover"', 'id="modal-upload"', 'id="modal-api"', 'id="ix-btn-eye"', 'id="ix-btn-random"'):
            self.assertIn(marker, html)
        self.assertIn('.v5-popover', css)
        self.assertIn('showPopover', js)

    def test_05_v5_interactive_curated_pool_and_observer_isolation(self):
        _, _, meta_raw = self._get('/api/meta')
        token = json.loads(meta_raw.decode('utf-8'))['scene_token']

        st, _, ix_raw = self._get('/api/v5/interactive/meta')
        self.assertEqual(st, 200)
        ix_meta = json.loads(ix_raw.decode('utf-8'))
        stats = ix_meta['stats']
        self.assertGreaterEqual(stats['aerial_images'], 190)
        self.assertGreaterEqual(stats['aerial_tasks'], 900)
        self.assertEqual(stats['ground_tasks'], 192)

        aerial_tasks = [t for t in ix_meta['tasks'] if t.get('dataset_id') == 'aerial_bank']
        per_img = Counter((t['bank_id'], t['area'], t['grid']) for t in aerial_tasks)
        self.assertLessEqual(max(per_img.values()), 5)

        sample_t = next(t for t in aerial_tasks if t['bank_id'] == 'swissview-1')
        st, sess = self._post('/api/v5/interactive/create', {
            'task_key': sample_t['task_key'],
            'policy': 'M0',
            'start': 0,
            'goal': 24,
            'seed': 0,
        }, token)
        self.assertEqual(st, 200)
        self.assertEqual(sess['visited'], [0])

        obs_st, obs_mime, obs_bytes = self._get(
            f"/api/v5/interactive/image?task_key={sample_t['task_key']}&cell=24&observer=1"
        )
        self.assertEqual(obs_st, 200)
        self.assertIn('image/png', obs_mime)
        self.assertGreater(len(obs_bytes), 200)

        _, _, st_raw = self._get('/api/v5/interactive/state')
        self.assertEqual(json.loads(st_raw.decode('utf-8'))['visited'], [0])

        st, stepped = self._post('/api/v5/interactive/step', {'action': None}, token)
        self.assertEqual(st, 200)
        self.assertEqual(stepped['step'], 1)
        self.assertEqual(len(stepped['decision']['probabilities']), 4)
        self.assertTrue(stepped['decision'].get('branch'))

        st, switched = self._post('/api/v5/interactive/mode', {'policy': 'frontier'}, token)
        self.assertEqual(st, 200)
        self.assertEqual(switched['policy'], 'frontier')

        st, rst = self._post('/api/v5/interactive/reset', {}, token)
        self.assertEqual(st, 200)
        self.assertTrue(rst['reset'])

    def test_06_v5_volatile_upload_and_cloud_api_config(self):
        _, _, meta_raw = self._get('/api/meta')
        token = json.loads(meta_raw.decode('utf-8'))['scene_token']

        im = Image.new('RGB', (480, 480), color=(70, 120, 175))
        draw = ImageDraw.Draw(im)
        draw.rectangle([90, 90, 260, 260], fill=(215, 185, 90))
        buf = BytesIO()
        im.save(buf, format='PNG')
        b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

        st, up = self._post('/api/v5/interactive/upload', {
            'image': b64,
            'name': 'V5临时3x3测试',
            'grid': 3,
        }, token)
        self.assertEqual(st, 200)
        self.assertEqual(up['grid'], 3)
        self.assertEqual(up['budget'], 6)

        st, s3 = self._post('/api/v5/interactive/create', {
            'task_key': up['task_key'],
            'policy': 'M0',
            'start': 0,
            'goal': 8,
            'seed': 0,
        }, token)
        self.assertEqual(st, 200)
        self.assertEqual(s3['grid'], 3)

        st, step_u = self._post('/api/v5/interactive/step', {'action': None}, token)
        self.assertEqual(st, 200)
        self.assertEqual(step_u['step'], 1)
        self.assertEqual(len(step_u['decision']['probabilities']), 4)

        st, deleted = self._post('/api/v5/interactive/upload/delete', {'task_key': up['task_key']}, token)
        self.assertEqual(st, 200)
        self.assertEqual(deleted['deleted'], up['task_key'])

        old_cfg = self.workbench.config
        old_mgr_cfg = self.server.interactive_mgr.wb.config
        try:
            st, cfg_pub = self._post('/api/v5/interactive/config', {
                'base_url': 'https://api.deepseek.com/v1',
                'model': 'deepseek-flash',
                'api_key': 'sk-test-v5-secret-9999',
                'timeout': 30,
                'disable_thinking': True,
            }, token)
            self.assertEqual(st, 200)
            self.assertEqual(cfg_pub['model'], 'deepseek-flash')
            self.assertTrue(cfg_pub['disable_thinking'])
            self.assertNotIn('sk-test-v5-secret-9999', json.dumps(cfg_pub))

            _, _, new_meta = self._get('/api/meta')
            parsed_meta = json.loads(new_meta.decode('utf-8'))
            self.assertEqual(parsed_meta['api']['model'], 'deepseek-flash')
            self.assertTrue(parsed_meta['api']['disable_thinking'])
        finally:
            self.workbench.config = old_cfg
            self.server.interactive_mgr.wb.config = old_mgr_cfg


if __name__ == '__main__':
    unittest.main()

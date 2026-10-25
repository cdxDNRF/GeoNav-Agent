"""CPU-only orchestration tests: no torch, models, navigation, or GPU probing."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'project/src/train/run_geoexplorer_training_v2.py'
spec = importlib.util.spec_from_file_location('training_driver_v2', SOURCE)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
TEMP = ROOT / 'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v2/核验'


class DriverTests(unittest.TestCase):
    def test_resource_wait_and_threshold(self):
        times = iter([0, 0, 15]); probes = iter([1000, 4560]); sleeps = []
        free, elapsed = driver.wait_for_gpu(4560, 600, lambda: next(probes), lambda: next(times), sleeps.append)
        self.assertEqual((free, elapsed), (4560, 15)); self.assertEqual(sleeps, [15])

    def test_no_resource_means_no_execution(self):
        with self.assertRaisesRegex(RuntimeError, 'resource unavailable'):
            driver.wait_for_gpu(4560, 0, lambda: 800, lambda: 0, lambda _: self.fail('must not sleep'))

    def test_probe_failure_propagates(self):
        with self.assertRaises(OSError):
            driver.wait_for_gpu(4560, 0, lambda: (_ for _ in ()).throw(OSError('missing GPU')))

    def test_no_retry_of_consumed_seed(self):
        with tempfile.TemporaryDirectory(prefix='测试临时_', dir=TEMP) as folder:
            path = Path(folder); (path / 'seed_0').mkdir()
            with self.assertRaises(FileExistsError): driver.ensure_fresh(path)

    def test_start_lock_blocks_second_driver(self):
        with tempfile.TemporaryDirectory(prefix='测试临时_', dir=TEMP) as folder:
            path = Path(folder); (path / '核验').mkdir(); (path / '核验/启动锁.json').write_text('{}')
            with self.assertRaises(FileExistsError): driver.ensure_fresh(path)

    def test_write_new_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix='测试临时_', dir=TEMP) as folder:
            path = Path(folder) / 'record.json'; driver.write_new(path, {'a': 1})
            with self.assertRaises(FileExistsError): driver.write_new(path, {'a': 2})
            self.assertEqual(json.loads(path.read_text()), {'a': 1})

    def test_complete_requires_exact_counts_and_protocol(self):
        result = dict(seed=0, status='completed', epoch=300, microbatches=32700,
                      pretrain_updates=300, environment_steps=163840, ppo_updates=256,
                      protocol_sha256='frozen', scientific_sr=False)
        driver.check_result(result, 0, 'frozen')
        for key, value in [('seed', 1), ('microbatches', 32701), ('environment_steps', 163839),
                           ('ppo_updates', 255), ('protocol_sha256', 'other'), ('scientific_sr', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.check_result(dict(result, **{key: value}), 0, 'frozen')

    def test_preflight_failure_has_terminal_record_and_no_child(self):
        with tempfile.TemporaryDirectory(prefix='测试临时_', dir=TEMP) as folder:
            batch = Path(folder); (batch / '核验').mkdir(); protocol_path = batch / 'protocol.json'
            protocol_path.write_text('{}')
            protocol = {'budget': {'seeds': [0, 1, 2], 'caps': {}}}
            authorization = {'resource_wait_seconds_total': 0, 'previous_failed_attempt': 'v1'}
            with patch.object(driver, 'validate_batch', return_value=(batch, protocol_path, protocol, authorization)), \
                 patch.object(driver, 'wait_for_gpu', side_effect=RuntimeError('GPU busy')), \
                 patch.object(driver.subprocess, 'Popen') as process:
                self.assertEqual(driver.main(batch), 1); process.assert_not_called()
            record = json.loads((batch / '训练结果汇总.json').read_text(encoding='utf-8'))
            self.assertEqual(record['status'], 'incomplete'); self.assertEqual(record['runs'][0]['status'], 'preflight')
            self.assertEqual(record['confirmation_navigation'], 0)

    def test_exit_does_not_signal_dead_process(self):
        from unittest.mock import Mock
        process = Mock(); process.poll.return_value = 0
        self.assertEqual(driver.terminate_owned(process), 'already_exited')
        process.send_signal.assert_not_called()

    def test_restart_cannot_use_original_batch(self):
        with self.assertRaisesRegex(ValueError, 'newly registered'):
            driver.validate_batch(ROOT / 'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v1')

    def test_authorization_and_protocol_binding_are_required(self):
        with tempfile.TemporaryDirectory(prefix='测试临时_', dir=TEMP) as folder:
            root = Path(folder); batch = root / 'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v2'
            (batch / '元数据').mkdir(parents=True)
            protocol_path = batch / '元数据/冻结执行协议.json'; protocol_path.write_text('{}')
            auth_path = batch / '元数据/重启执行登记.json'
            auth_path.write_text(json.dumps({'user_authorized_restart': False}))
            with patch.object(driver, 'ROOT', root), self.assertRaisesRegex(ValueError, 'authorization missing'):
                driver.validate_batch(batch)
            auth_path.write_text(json.dumps({'user_authorized_restart': True, 'protocol_sha256': 'other'}))
            with patch.object(driver, 'ROOT', root), self.assertRaisesRegex(ValueError, 'does not bind'):
                driver.validate_batch(batch)

    def test_three_successful_children_are_sequential_and_recorded(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(prefix='测试临时_', dir=TEMP) as folder:
            batch = Path(folder); (batch / '核验').mkdir(); protocol_path = batch / 'protocol.json'
            protocol_path.write_text('{}')
            protocol = {'budget': {'seeds': [0, 1, 2], 'caps': {'wall_seconds_total': 64800,
                         'wall_seconds_per_seed': 21600, 'output_bytes_total': 18 * 2**30}}}
            authorization = {'resource_wait_seconds_total': 0, 'previous_failed_attempt': 'v1'}
            consumed = []
            def start(command, **kwargs):
                seed = int(command[-1]); consumed.append(seed)
                seed_dir = batch / f'seed_{seed}'; seed_dir.mkdir()
                driver.write_new(seed_dir / 'result.json', dict(seed=seed, status='completed', epoch=300,
                    microbatches=32700, pretrain_updates=300, environment_steps=163840, ppo_updates=256,
                    protocol_sha256=driver.sha(protocol_path), scientific_sr=False, elapsed_seconds=1))
                process = Mock(); process.pid = 100 + seed; process.wait.return_value = 0
                process.poll.return_value = 0
                return process
            with patch.object(driver, 'validate_batch', return_value=(batch, protocol_path, protocol, authorization)), \
                 patch.object(driver, 'wait_for_gpu', return_value=(5000, 0)), \
                 patch.object(driver.subprocess, 'Popen', side_effect=start):
                self.assertEqual(driver.main(batch), 0)
            summary = json.loads((batch / '训练结果汇总.json').read_text(encoding='utf-8'))
            self.assertEqual(consumed, [0, 1, 2]); self.assertEqual(summary['status'], 'completed')
            self.assertTrue(all((batch / f'核验/seed_{s}_退出.json').exists() for s in range(3)))


if __name__ == '__main__':
    unittest.main()

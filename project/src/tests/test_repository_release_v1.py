"""Publication boundary checks, no real dataset or model imports."""
from pathlib import Path
import tempfile
import unittest
import subprocess

try:
    from project.src.tools import repository_release_v1 as release
except ImportError:
    release = None


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(release, "发布核验工具尚未实现")
        self.repo_root = Path(__file__).resolve().parents[3]
        self.registered_parent = self.repo_root / '.agents/本地/整理记录_20261005_v1/核验'
        self.temp_parent = self.registered_parent

    def temporary(self):
        parent = self.temp_parent.resolve()
        if not parent.is_relative_to(self.repo_root) or not parent.is_relative_to(self.registered_parent.resolve()):
            raise ValueError('temporary parent must remain within the registered test directory')
        parent.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='测试临时_', dir=parent)

    def test_local_and_bulk_data_are_rejected_but_summaries_allowed(self):
        denied = [".agents/本地/提示.md", "project/src/tools/local/archive/pi.py", "DATA/raw_data/x.jpg",
                  "DATA/processed_data/X/主对照/a.jsonl", "DATA/processed_data/X/工程数据/sources/a.tif", "models/a.safetensors"]
        self.assertEqual({x['path'] for x in release.publication_issues(denied)}, set(denied))
        self.assertEqual(release.publication_issues([".agents/TASKS.md", "docs/PROJECT_STATE.md", "project/.env.example", "DATA/processed_data/X/主对照/汇总.json", "project/src/webapp/server_v1.py", "DATA/processed_data/Masa/episodes/episodes_val.jsonl", "DATA/processed_data/Masa/评测结果/S2稳定验证_v1/固定任务.jsonl"]), [])

    def test_traversal_and_actual_env_are_rejected(self):
        paths = ["../a", "project/.env", "project/.env.deepseek", ".git/config"]
        self.assertEqual(len(release.publication_issues(paths)), 4)

    def test_relocation_preserves_identity_and_rejects_escape(self):
        with self.temporary() as folder:
            root = Path(folder); (root/'archived').mkdir(); (root/'archived/doc.md').write_text('original',encoding='utf-8')
            result = release.resolve_moved(root, 'docs/doc.md', {'docs/doc.md':'archived/doc.md'})
            self.assertEqual(result.read_text(encoding='utf-8'), 'original')
            with self.assertRaises(ValueError):
                release.resolve_moved(root, 'docs/doc.md', {'docs/doc.md':'../escape'})
            with self.assertRaises(ValueError):
                release.resolve_moved(root, str(root/'archived/doc.md'), {})

    def test_existing_original_is_not_hidden_by_mapping(self):
        with self.temporary() as folder:
            root=Path(folder);(root/'old.txt').write_text('changed');(root/'new.txt').write_text('original')
            self.assertEqual(release.resolve_moved(root,'old.txt',{'old.txt':'new.txt'}),root/'old.txt')

    def test_secret_detector_does_not_return_secret_value(self):
        token='sk-'+'X'*40
        result=release.secret_locations('api_key = "'+token+'"')
        self.assertTrue(result)
        self.assertNotIn(token, str(result))
        self.assertFalse(release.secret_locations('api_key = "not-needed"'))

    def test_seal_hash_mismatch_raises(self):
        with self.temporary() as folder:
            root=Path(folder);(root/'x.txt').write_text('wrong')
            with self.assertRaises(ValueError):
                release.verify_hashes(root, {'x.txt':'0'*64}, {})

    def test_temporary_parent_can_be_missing_in_a_clone(self):
        with self.temporary() as holder:
            self.temp_parent = Path(holder) / 'missing_parent'
            self.assertFalse(self.temp_parent.exists())
            with self.temporary() as generated:
                self.assertTrue(Path(generated).is_relative_to(self.temp_parent))

    def test_inspection_reads_index_instead_of_working_tree(self):
        with self.temporary() as folder:
            root = Path(folder)
            subprocess.run(['git', 'init', '-q', folder], check=True)
            path = root / 'safe.py'
            path.write_text('x = 1\n')
            subprocess.run(['git', 'add', 'safe.py'], cwd=root, check=True)
            path.write_text('api_key = "' + 'sk-' + 'X'*40 + '"\n')
            self.assertTrue(release.inspect_index(root)['ok'])
            subprocess.run(['git', 'add', 'safe.py'], cwd=root, check=True)
            result = release.inspect_index(root)
            self.assertFalse(result['ok'])
            self.assertEqual(result['issues'][0]['reason'], 'credential_match')

    def test_new_size_limit_and_syntax_are_checked(self):
        with self.temporary() as folder:
            root = Path(folder)
            subprocess.run(['git', 'init', '-q', folder], check=True)
            (root/'bad.py').write_text('this is invalid syntax\n')
            subprocess.run(['git', 'add', 'bad.py'], cwd=root, check=True)
            result = release.inspect_index(root, max_new_bytes=10)
            self.assertEqual({x['reason'] for x in result['issues']}, {'new_file_too_large', 'python_syntax'})


if __name__=='__main__':
    unittest.main()

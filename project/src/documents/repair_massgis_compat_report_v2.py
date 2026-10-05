"""Repair Markdown table spacing without overwriting DATA-007 v1 evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'DATA/processed_data/MassGIS/工程准备/同产品开发导航兼容_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    src = OUT/'同产品开发导航兼容报告.md'
    dst = OUT/'同产品开发导航兼容报告_v2.md'
    note = OUT/'核验/报告排版修正_v2.json'
    if dst.exists() or note.exists():
        raise ValueError('No overwrite of report repair')
    binding = json.loads((OUT/'核验/交付生成绑定.json').read_text(encoding='utf-8'))
    if sha(src) != binding['output_sha256'][src.name]:
        raise ValueError('Original report changed')
    lines = src.read_text(encoding='utf-8').splitlines()
    fixed = []
    for i, line in enumerate(lines):
        if not line and fixed and fixed[-1].startswith('|'):
            j = i+1
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines) and lines[j].startswith('|'):
                continue
        fixed.append(line)
    text = '\n'.join(fixed)+'\n'
    tail = ('\n实际10×10 Coverage3Radial线索接受227次，其中153次直接命中真实目标、74次未命中；'
            '67.40%的导航接受精度低于raw图对精度。实际路径、候选先验及原合法/未访问门控会改变接受分布，'
            '原因尚未逐项验证；不能以raw门槛通过宣称实际接受也达到90%。下一轮协议应单列这项风险。\n')
    with dst.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text+tail)
    record = dict(original_report_sha256=sha(src), original_binding_sha256=sha(OUT/'核验/交付生成绑定.json'),
        source_sha256={Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
        output_sha256={dst.name: sha(dst)}, changes='Remove blank inside table; clarify saved navigation acceptance risk',
        new_model_calls=0, new_navigation=0, original_changed=False,
        data_sha256=sha(OUT/'兼容补充分层结果.json'))
    with note.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, ensure_ascii=False, indent=2); f.write('\n')
    print('Report v2 published; original report/binding preserved; metrics unchanged.')


if __name__ == '__main__':
    main()

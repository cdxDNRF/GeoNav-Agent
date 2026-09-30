"""Run through browser-harness stdin (its helpers are pre-imported).
browser-harness < project/src/documents/render_report_figure.py
Requires the dedicated automation browser; never previews the PNG.
"""
from pathlib import Path
import base64
ROOT=Path(r'D:/桌面/XDUCS-courses/大四/大数据工程')
p=ROOT/'project/src/documents/figure_tech_route_corrected.html'
new_tab(p.as_uri())
wait_for_load()
cdp('Emulation.setDeviceMetricsOverride',width=1240,height=920,deviceScaleFactor=2,mobile=False)
js('document.fonts.ready.then(() => true)')
box=js("(() => {const r=document.querySelector('.diagram-canvas').getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height,scale:1};})()")
res=cdp('Page.captureScreenshot',format='png',clip=box,captureBeyondViewport=True)
out=ROOT/'选题报告相关/校正审阅/figure_tech_route_校正版.png'
out.write_bytes(base64.b64decode(res['data']))
print('Saved:',out,box)

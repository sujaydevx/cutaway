function serialize(svg: SVGSVGElement): { xml: string; w: number; h: number } {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const vb = svg.viewBox.baseVal;
  clone.setAttribute('width', String(vb.width));
  clone.setAttribute('height', String(vb.height));
  clone.removeAttribute('style');
  const xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(clone);
  return { xml, w: vb.width, h: vb.height };
}

function save(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

export function downloadSvg(svg: SVGSVGElement | null, name: string) {
  if (!svg) return;
  save(new Blob([serialize(svg).xml], { type: 'image/svg+xml' }), name + '.svg');
}

export function downloadPng(svg: SVGSVGElement | null, name: string) {
  if (!svg) return;
  const { xml, w, h } = serialize(svg);
  const url = URL.createObjectURL(new Blob([xml], { type: 'image/svg+xml;charset=utf-8' }));
  const img = new Image();
  img.onload = () => {
    const scale = 3;
    const c = document.createElement('canvas');
    c.width = w * scale;
    c.height = h * scale;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#FBFCFD';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(url);
    c.toBlob((b) => b && save(b, name + '.png'));
  };
  img.src = url;
}

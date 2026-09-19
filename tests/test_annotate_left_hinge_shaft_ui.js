#!/usr/bin/gjs
/* Smoke-test the embedded canvas interaction without a GUI. */
const {GLib} = imports.gi;

let sourcePath = ARGV[0] || 'g2_adapter/annotate_left_hinge_shaft_sam3.py';
let [ok, bytes] = GLib.file_get_contents(sourcePath);
if (!ok) throw new Error('cannot read ' + sourcePath);
let source = imports.byteArray.toString(bytes);
let match = source.match(/PAGE = r"""([\s\S]*?)"""/);
if (!match) throw new Error('PAGE not found');
let script = match[1].match(/<script>([\s\S]*)<\/script>/)[1];
/* Do not perform the real initial fetch/image load in this unit test. */
script = script.replace(/document\.addEventListener\('keydown',[\s\S]*?\);load\(\);\s*$/, '');

let handlers = {};
let elements = {};
function element(id) {
    if (!elements[id]) elements[id] = {
        classList: {add() {}, remove() {}, toggle() {}},
        textContent: '', value: '', max: 0,
    };
    return elements[id];
}
let strokeWidths = [], fillCount = 0;
let ctx = {
    clearRect() {}, drawImage() {}, beginPath() {}, ellipse() {}, stroke() {},
    arc() {}, fill() {}, moveTo() {}, lineTo() {}, closePath() {},
    strokeStyle: '', fillStyle: '', lineWidth: 0,
};
ctx.stroke = function() { strokeWidths.push(this.lineWidth); };
ctx.fill = function() { fillCount++; };
let canvas = {
    width: 400, height: 300,
    addEventListener(name, fn) { handlers[name] = fn; },
    setPointerCapture() {}, hasPointerCapture() { return false; }, releasePointerCapture() {},
    getBoundingClientRect() { return {left: 0, top: 0, width: 400, height: 300}; },
    getContext() { return ctx; },
};
let fakeDocument = {
    querySelector(sel) { return sel === '#canvas' ? canvas : element(sel.slice(1)); },
    addEventListener() {},
};
let fakeImage = function() { this.complete = true; };
let fakeWindow = {document: fakeDocument, Image: fakeImage};
/* Make browser globals visible to the embedded page code. */
var document = fakeDocument;
var Image = fakeImage;
eval('(function(){' + script + '\n})();');

/* The page code is scoped by the function above, so run the same event path
 * in a second eval with assertions in that scope. */
let testScript = script + `
function assert(c, m) { if (!c) throw new Error(m); }
width=400; height=300; c.width=400; c.height=300; img.complete=true;
ellipseMode=true; currentClass=1;
handlers.pointerdown({button:0, pointerId:1, clientX:100, clientY:100, preventDefault(){}});
handlers.pointermove({pointerId:1, clientX:220, clientY:180});
handlers.pointerup({button:0, pointerId:1});
assert(polys.length===1 && polys[0].shape==='ellipse', 'ellipse was not created');
assert(ellipseMode===false && selected===0, 'create mode did not switch to edit mode');
let oldCx=polys[0].cx, oldCy=polys[0].cy;
handlers.pointerdown({button:0, pointerId:2, clientX:160, clientY:140, preventDefault(){}});
handlers.pointermove({pointerId:2, clientX:180, clientY:155});
handlers.pointerup({button:0, pointerId:2});
assert(polys[0].cx===oldCx+20 && polys[0].cy===oldCy+15, 'ellipse center did not move');
let oldRx=polys[0].rx;
handlers.pointerdown({button:0, pointerId:3, clientX:polys[0].cx+oldRx, clientY:polys[0].cy, preventDefault(){}});
handlers.pointermove({pointerId:3, clientX:polys[0].cx+oldRx+20, clientY:polys[0].cy});
handlers.pointerup({button:0, pointerId:3});
assert(polys[0].rx===oldRx+20, 'ellipse handle did not resize');
let saved=serializable()[0];
assert(saved.shape==='ellipse' && saved.points.length===48, 'ellipse serialization is not editable metadata');
width=1280; height=1056; c.width=1280; c.height=1056;
polys=[{class_id:1,shape:'ellipse',cx:640,cy:528,rx:20,ry:16}];
selected=-1; strokeWidths=[]; draw();
assert(strokeWidths[0]===2, 'unselected hole outline must be 2 px');
selected=0; strokeWidths=[]; draw();
assert(strokeWidths[0]===2.5, 'selected hole outline must be 2.5 px');
polys=[{class_id:1,points:[[620,510],[660,510],[660,546],[620,546]]}];
selected=-1; fillCount=0; draw();
assert(fillCount===0, 'saved hole polygon must not draw dense vertex dots');
print('UI ellipse create/edit/serialize smoke test: PASS');
`;
eval('(function(){' + testScript + '\n})();');

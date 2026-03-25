"""
FaceTransform DSP - Facial Image Warping, Aging & Expression Transformation
CENG 384 - DSP Project
"""
import os, io, csv, base64, uuid
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from flask import Flask, render_template, request, jsonify, send_file
from scipy.fft import fft2, ifft2, fftshift
from skimage.metrics import structural_similarity as compare_ssim

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)

MODEL_PATH = os.path.join(BASE, 'face_landmarker.task')
STD_SIZE = (512, 512)

# ─── Helpers ───
def enc(img):
    ok, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode() if ok else ''

def dec(b64):
    if ',' in b64: b64 = b64.split(',',1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)

# ─── FR-3.1 Preprocessing ───
def preprocess(img):
    if img is None: return None, 'Bad image'
    h, w = img.shape[:2]
    if h < 50 or w < 50: return None, 'Too small'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cas.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    if len(faces) > 0:
        x,y,fw,fh = faces[0]
        pad = int(max(fw,fh)*0.4)
        x1,y1 = max(0,x-pad), max(0,y-pad)
        x2,y2 = min(w,x+fw+pad), min(h,y+fh+pad)
        img = img[y1:y2, x1:x2]
    return cv2.resize(img, STD_SIZE, interpolation=cv2.INTER_AREA), None

# ─── FR-3.2 Landmark Detection ───
def detect_landmarks(img):
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
    pts = []
    with FaceLandmarker.create_from_options(opts) as det:
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.face_landmarks:
            for lm in res.face_landmarks[0]:
                pts.append([int(lm.x*w), int(lm.y*h)])
    return pts

def draw_lm(img, pts):
    out = img.copy()
    for p in pts: cv2.circle(out, tuple(p), 1, (0,255,0), -1)
    return out

# ─── FR-3.3 Expression Warping ───
MOUTH_CORNERS = [61, 291]
MOUTH_TOP = [13]; MOUTH_BOTTOM = [14]
UPPER_LIP = [61,185,40,39,37,0,267,269,270,409,291]
LOWER_LIP = [61,146,91,181,84,17,314,405,321,375,291]
LEFT_BROW = [70,63,105,66,107,55,65,52,53,46]
RIGHT_BROW = [300,293,334,296,336,285,295,282,283,276]
JAW = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
       400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

def _warp_tri(src_img, dst_img, ts, td):
    rs = cv2.boundingRect(np.float32([ts]))
    rd = cv2.boundingRect(np.float32([td]))
    tsr = [(ts[i][0]-rs[0], ts[i][1]-rs[1]) for i in range(3)]
    tdr = [(td[i][0]-rd[0], td[i][1]-rd[1]) for i in range(3)]
    crop = src_img[rs[1]:rs[1]+rs[3], rs[0]:rs[0]+rs[2]]
    if crop.size==0 or rd[2]<=0 or rd[3]<=0: return
    M = cv2.getAffineTransform(np.float32(tsr), np.float32(tdr))
    warped = cv2.warpAffine(crop, M, (rd[2],rd[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((rd[3],rd[2],3), np.float32)
    cv2.fillConvexPoly(mask, np.int32(tdr), (1,1,1), cv2.LINE_AA)
    h_i,w_i = dst_img.shape[:2]
    y1,y2,x1,x2 = rd[1], rd[1]+rd[3], rd[0], rd[0]+rd[2]
    if y2>h_i or x2>w_i or y1<0 or x1<0: return
    reg = dst_img[y1:y2, x1:x2].astype(np.float32)
    dst_img[y1:y2, x1:x2] = (reg*(1-mask) + warped.astype(np.float32)*mask).astype(np.uint8)

def warp_face(img, src, dst):
    h, w = img.shape[:2]
    out = img.copy()
    bnd = [(0,0),(w//2,0),(w-1,0),(0,h//2),(w-1,h//2),(0,h-1),(w//2,h-1),(w-1,h-1)]
    a_s = list(src) + bnd
    a_d = list(dst) + bnd
    sub = cv2.Subdiv2D((0,0,w,h))
    idx_map = {}
    for i,p in enumerate(a_d):
        px = (max(0,min(w-1,int(p[0]))), max(0,min(h-1,int(p[1]))))
        idx_map[px] = i
        sub.insert((float(px[0]), float(px[1])))
    for tri in sub.getTriangleList():
        ps = [(int(tri[j*2]),int(tri[j*2+1])) for j in range(3)]
        if not all(0<=p[0]<w and 0<=p[1]<h for p in ps): continue
        idx = []
        for p in ps:
            if p in idx_map: idx.append(idx_map[p])
            else:
                best = min(range(len(a_d)), key=lambda j:(a_d[j][0]-p[0])**2+(a_d[j][1]-p[1])**2)
                idx.append(best)
        if len(set(idx))<3: continue
        try: _warp_tri(img, out, [a_s[i] for i in idx], [a_d[i] for i in idx])
        except: continue
    return out

def apply_expression(img, lm, etype, intensity):
    if not lm or len(lm)<300: return img
    h, w = img.shape[:2]
    f = intensity / 100.0
    sp, dp = [], []
    if etype == 'smile':
        idxs = list(set(MOUTH_CORNERS+MOUTH_TOP+MOUTH_BOTTOM+UPPER_LIP+LOWER_LIP))
        for i in idxs:
            if i>=len(lm): continue
            sx,sy = lm[i]
            sp.append((sx,sy))
            cx = w//2
            if i in MOUTH_CORNERS:
                dx = 1 if sx>cx else -1
                dp.append((sx+int(dx*6*f), sy-int(10*f)))
            elif i in MOUTH_TOP:
                dp.append((sx, sy-int(3*f)))
            elif i in MOUTH_BOTTOM:
                dp.append((sx, sy+int(3*f)))
            else:
                dx = 1 if sx>cx else -1
                dp.append((sx+int(dx*2*f), sy-int(2*f)))
    elif etype == 'eyebrow':
        for i in LEFT_BROW+RIGHT_BROW:
            if i>=len(lm): continue
            sx,sy = lm[i]
            sp.append((sx,sy)); dp.append((sx, sy-int(12*f)))
    elif etype == 'lips':
        for i in list(set(UPPER_LIP+LOWER_LIP)):
            if i>=len(lm): continue
            sx,sy = lm[i]
            cx = w//2
            dx = 1 if sx>cx else -1
            sp.append((sx,sy)); dp.append((sx+int(dx*8*f), sy))
    elif etype == 'slim':
        for i in JAW:
            if i>=len(lm): continue
            sx,sy = lm[i]
            cx = w//2
            dx = -1 if sx>cx else 1
            off = int(abs(sx-cx)*0.1*f)
            sp.append((sx,sy)); dp.append((sx+dx*off, sy))
    if len(sp)<4: return img
    result = warp_face(img, sp, dp)
    # Smooth blend
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.array(dp, np.int32)), 255)
    mask = cv2.GaussianBlur(mask, (31,31), 15)
    m3 = cv2.merge([mask]*3).astype(np.float32)/255
    return np.clip(img.astype(np.float32)*(1-m3)+result.astype(np.float32)*m3, 0, 255).astype(np.uint8)

# ─── FR-3.4 Aging / De-Aging ───
def apply_aging(img, intensity, mode='aging'):
    h, w = img.shape[:2]
    f = intensity / 100.0
    channels = []
    for c in range(3):
        ch = img[:,:,c].astype(np.float64)
        F = fftshift(fft2(ch))
        rows, cols = ch.shape
        cr, cc = rows//2, cols//2
        mr = np.sqrt(cr**2+cc**2)
        Y, X = np.ogrid[:rows,:cols]
        dist = np.sqrt((X-cc)**2+(Y-cr)**2)
        if mode == 'aging':
            cutoff = mr*0.15
            hp = 1.0/(1.0+(cutoff/(dist+1e-10))**4)
            F = F * (1.0 + hp*f*1.5)
        else:
            cutoff = max(mr*(0.25-0.1*f), mr*0.1)
            lp = 1.0/(1.0+(dist/(cutoff+1e-10))**4)
            lp = np.clip(lp+0.15, 0, 1)
            F = F * lp
        back = np.clip(np.real(ifft2(fftshift(F))), 0, 255).astype(np.uint8)
        channels.append(back)
    result = cv2.merge(channels)
    if mode == 'aging':
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=1.5+f*2, tileGridSize=(8,8)).apply(lab[:,:,0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float64)
        result[:,:,0] *= (1-0.04*f)
        result[:,:,2] *= (1+0.03*f)
        result = np.clip(result, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.GaussianBlur(cv2.Canny(gray,50,120).astype(np.float64)/255, (5,5), 1.5)
        noise = np.random.normal(0, 8*f, (h,w)) * edges
        for ci in range(3):
            result[:,:,ci] = np.clip(result[:,:,ci].astype(np.float64)+noise, 0, 255).astype(np.uint8)
    else:
        s = max(int(30*f), 3)
        result = cv2.bilateralFilter(result, 9, s, s)
        result = cv2.convertScaleAbs(result, alpha=1+0.05*f, beta=5*f)
    return result

# ─── FR-3.5 Frequency Analysis ───
def freq_analysis(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    F = fftshift(fft2(gray))
    mag = np.log1p(np.abs(F))
    mag_n = ((mag-mag.min())/(mag.max()-mag.min()+1e-10)*255).astype(np.uint8)
    vis = cv2.applyColorMap(mag_n, cv2.COLORMAP_JET)
    power = np.abs(F)**2
    total = float(np.sum(power))
    rows,cols = gray.shape
    Y,X = np.ogrid[:rows,:cols]
    dist = np.sqrt((X-cols//2)**2+(Y-rows//2)**2)
    low_mask = dist <= min(rows,cols)//8
    low = float(np.sum(power[low_mask]))
    high = float(np.sum(power[~low_mask]))
    return dict(spectrum=enc(vis), total=total, low=low, high=high,
                ratio=low/(high+1e-10), low_pct=low/(total+1e-10)*100, high_pct=high/(total+1e-10)*100)

# ─── FR-3.6 Metrics ───
def metrics(o, t):
    g1 = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    if g1.shape!=g2.shape: g2=cv2.resize(g2,(g1.shape[1],g1.shape[0]))
    mse = float(np.mean((g1.astype(np.float64)-g2.astype(np.float64))**2))
    psnr = 100.0 if mse<1e-10 else float(10*np.log10(255**2/mse))
    ss = float(compare_ssim(g1, g2))
    return dict(mse=round(mse,2), psnr=round(psnr,2), ssim=round(ss,4))

# ─── Routes ───
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'image' not in request.files: return jsonify(error='No file'), 400
    f = request.files['image']
    if not f.filename: return jsonify(error='No file'), 400
    ext = f.filename.rsplit('.',1)[-1].lower() if '.' in f.filename else ''
    if ext not in ('jpg','jpeg','png'): return jsonify(error='JPG/PNG only'), 400
    raw = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None: return jsonify(error='Bad image'), 400
    proc, err = preprocess(img)
    if err: return jsonify(error=err), 400
    lm = detect_landmarks(proc)
    return jsonify(image=enc(proc), landmarks=lm, count=len(lm))

@app.route('/api/landmarks_overlay', methods=['POST'])
def api_lm():
    d = request.get_json()
    img = dec(d['image'])
    pts = [tuple(p) for p in d['landmarks']]
    out = draw_lm(img, pts) if d.get('show', True) else img
    return jsonify(image=enc(out))

@app.route('/api/export_landmarks', methods=['POST'])
def api_export_lm():
    d = request.get_json()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['Index','X','Y'])
    for i,p in enumerate(d.get('landmarks',[])): w.writerow([i,p[0],p[1]])
    mem = io.BytesIO(buf.getvalue().encode()); mem.seek(0)
    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='landmarks.csv')

@app.route('/api/expression', methods=['POST'])
def api_expr():
    d = request.get_json()
    img = dec(d['image'])
    lm = [list(p) for p in d['landmarks']]
    result = apply_expression(img, lm, d.get('type','smile'), int(d.get('intensity',50)))
    return jsonify(result=enc(result), metrics=metrics(img, result))

@app.route('/api/aging', methods=['POST'])
def api_aging():
    d = request.get_json()
    img = dec(d['image'])
    result = apply_aging(img, int(d.get('intensity',50)), d.get('mode','aging'))
    return jsonify(result=enc(result), metrics=metrics(img, result))

@app.route('/api/frequency', methods=['POST'])
def api_freq():
    d = request.get_json()
    img = dec(d['image'])
    r = freq_analysis(img)
    if d.get('transformed'):
        t = freq_analysis(dec(d['transformed']))
        r['t_spectrum']=t['spectrum']; r['t_total']=t['total']; r['t_low']=t['low']
        r['t_high']=t['high']; r['t_ratio']=t['ratio']
        r['t_low_pct']=t['low_pct']; r['t_high_pct']=t['high_pct']
    return jsonify(**r)

@app.route('/api/export_metrics', methods=['POST'])
def api_export_m():
    d = request.get_json()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['Transformation','MSE','PSNR','SSIM'])
    for r in d.get('rows',[]): w.writerow([r.get('name',''),r.get('mse',''),r.get('psnr',''),r.get('ssim','')])
    mem = io.BytesIO(buf.getvalue().encode()); mem.seek(0)
    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='metrics.csv')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

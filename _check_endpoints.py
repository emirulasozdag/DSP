import app, io, numpy as np, cv2

client = app.app.test_client()

img = np.zeros((256,256,3), dtype=np.uint8)
cv2.ellipse(img,(128,140),(70,90),0,0,360,(180,160,140),-1)
ok, buf = cv2.imencode('.jpg', img)
jpeg = buf.tobytes()

# health
r = client.get('/api/health')
print('GET /api/health:', r.status_code, r.get_json())

# preprocess
r = client.post('/api/preprocess',
    data={'image': (io.BytesIO(jpeg), 'face.jpg'), 'size': '128'},
    content_type='multipart/form-data')
print('POST /api/preprocess:', r.status_code, r.get_json())

# landmarks
r = client.post('/api/landmarks',
    data={'image': (io.BytesIO(jpeg), 'face.jpg')},
    content_type='multipart/form-data')
j = r.get_json()
slim = {k: v for k, v in j.items() if k not in ('landmarks','rawLandmarks','reduced68')}
print('POST /api/landmarks:', r.status_code, slim)

# transform zeros
r = client.post('/api/transform',
    data={'image': (io.BytesIO(jpeg), 'face.jpg'), 'smile':'0', 'brow':'0', 'lip':'0', 'slim':'0'},
    content_type='multipart/form-data')
j = r.get_json()
print('POST /api/transform (zeros):', r.status_code, {k:v for k,v in j.items() if k != 'image'})

# transform non-zero
r = client.post('/api/transform',
    data={'image': (io.BytesIO(jpeg), 'face.jpg'), 'smile':'50', 'brow':'30'},
    content_type='multipart/form-data')
j = r.get_json()
print('POST /api/transform (smile50):', r.status_code, {k:v for k,v in j.items() if k not in ('image','debug')})

# analysis missing transformed
r = client.post('/api/analysis',
    data={'image': (io.BytesIO(jpeg), 'face.jpg')},
    content_type='multipart/form-data')
print('POST /api/analysis (no transformed):', r.status_code, r.get_json())

# aging valid
r = client.post('/api/aging',
    data={'image': (io.BytesIO(jpeg), 'face.jpg'), 'mode':'aging', 'method':'freq',
          'years':'20', 'strength':'0.7', 'deterministic':'true'},
    content_type='multipart/form-data')
j = r.get_json()
print('POST /api/aging:', r.status_code, {k:v for k,v in j.items() if k not in ('image','variants')})

# aging bad mode
r = client.post('/api/aging',
    data={'image': (io.BytesIO(jpeg), 'face.jpg'), 'mode':'older'},
    content_type='multipart/form-data')
print('POST /api/aging bad mode:', r.status_code, r.get_json())

# aging bad hpf
r = client.post('/api/aging',
    data={'image': (io.BytesIO(jpeg), 'face.jpg'), 'hpf':'sharpen'},
    content_type='multipart/form-data')
print('POST /api/aging bad hpf:', r.status_code, r.get_json())

# transform: TPS mode
r = client.post('/api/transform',
    data={'image': (io.BytesIO(jpeg), 'face.jpg'), 'smile':'50', 'mode':'tps'},
    content_type='multipart/form-data')
j = r.get_json()
print('POST /api/transform (tps):', r.status_code, {k:v for k,v in j.items() if k not in ('image','debug')})

# deaging
r = client.post('/api/aging',
    data={'image': (io.BytesIO(jpeg), 'face.jpg'), 'mode':'deaging', 'method':'freq',
          'years':'20', 'strength':'0.7', 'deterministic':'true'},
    content_type='multipart/form-data')
j = r.get_json()
print('POST /api/deaging:', r.status_code, {k:v for k,v in j.items() if k not in ('image','variants')})

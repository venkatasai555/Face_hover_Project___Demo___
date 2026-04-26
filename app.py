import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import base64
import json
from io import BytesIO

st.set_page_config(layout="wide", page_title="Face Part Detector")

st.markdown("""
<style>
    .main { background: #0a0a0f; }
    .stApp { background: #0a0a0f; }
    h1 { 
        color: #fff; 
        font-family: 'Georgia', serif;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.title("✦ Face Feature Detector")
st.caption("Upload a face image — hover your mouse over any facial feature to identify it instantly")

mp_face_mesh = mp.solutions.face_mesh

def get_ids(connection_set):
    ids = set()
    for a, b in connection_set:
        ids.add(a)
        ids.add(b)
    return ids

LEFT_EYE_IDS   = get_ids(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDS  = get_ids(mp_face_mesh.FACEMESH_RIGHT_EYE)
LEFT_BROW_IDS  = get_ids(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
RIGHT_BROW_IDS = get_ids(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
LIPS_IDS       = get_ids(mp_face_mesh.FACEMESH_LIPS)
FACE_OVAL_IDS  = get_ids(mp_face_mesh.FACEMESH_FACE_OVAL)

NOSE_IDS = {
    1,2,3,4,5,6,19,20,44,45,48,49,51,64,75,79,
    94,97,98,99,115,125,126,129,131,141,142,166,
    195,197,198,209,217,218,219,220,235,236,237,
    238,239,240,241,242,243,274,275,278,279,294,
    326,327,358,360,370,392,399,412,419,420,432,
    437,438,439,440,457,458,459,460,461,462,463
}

RIGHT_IRIS_IDS = {468,469,470,471,472}
LEFT_IRIS_IDS  = {473,474,475,476,477}

def build_label_map():
    id_map = {}
    for lid in FACE_OVAL_IDS:  id_map[lid] = "Face"
    for lid in NOSE_IDS:       id_map[lid] = "Nose"
    for lid in LIPS_IDS:       id_map[lid] = "Lips"
    for lid in LEFT_BROW_IDS:  id_map[lid] = "Left Eyebrow"
    for lid in RIGHT_BROW_IDS: id_map[lid] = "Right Eyebrow"
    for lid in LEFT_EYE_IDS:   id_map[lid] = "Left Eye"
    for lid in RIGHT_EYE_IDS:  id_map[lid] = "Right Eye"
    for lid in LEFT_IRIS_IDS:  id_map[lid] = "Left Iris"
    for lid in RIGHT_IRIS_IDS: id_map[lid] = "Right Iris"
    return id_map

ID_TO_LABEL = build_label_map()

uploaded_file = st.file_uploader("Upload Face Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image   = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    h, w    = img_rgb.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    ) as face_mesh:
        results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        st.error("❌ No face detected. Please try a clear front-facing photo.")
        st.stop()

    # Collect landmarks
    raw_landmarks = []
    for face_lm in results.multi_face_landmarks:
        for idx, lm in enumerate(face_lm.landmark):
            px = int(lm.x * w)
            py = int(lm.y * h)
            raw_landmarks.append((idx, px, py))

    # Face bounding box for cheek/jaw/forehead classification
    xs = [lx for (_, lx, _) in raw_landmarks]
    ys = [ly for (_, _, ly) in raw_landmarks]
    face_top    = min(ys)
    face_bottom = max(ys)
    face_cx     = (min(xs) + max(xs)) / 2
    face_h      = face_bottom - face_top

    def classify(idx, px, py):
        base = ID_TO_LABEL.get(idx, "Face")
        if base != "Face":
            return base
        rel_y = (py - face_top) / face_h
        if rel_y > 0.72:
            return "Jawline"
        elif rel_y < 0.22:
            return "Forehead"
        elif px < face_cx:
            return "Right Cheek"
        else:
            return "Left Cheek"

    # Build landmark list for JS
    js_landmarks = []
    for (idx, px, py) in raw_landmarks:
        label = classify(idx, px, py)
        js_landmarks.append({"x": px, "y": py, "label": label})

    # Encode clean image to base64 (no dots)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    landmarks_json = json.dumps(js_landmarks)

    EMOJI = {
        "Left Eye":      "👁️ Left Eye",
        "Right Eye":     "👁️ Right Eye",
        "Left Iris":     "🔵 Left Iris",
        "Right Iris":    "🔵 Right Iris",
        "Left Eyebrow":  "✨ Left Eyebrow",
        "Right Eyebrow": "✨ Right Eyebrow",
        "Lips":          "👄 Lips",
        "Nose":          "👃 Nose",
        "Left Cheek":    "🫙 Left Cheek",
        "Right Cheek":   "🫙 Right Cheek",
        "Jawline":       "🫦 Jawline",
        "Forehead":      "🧠 Forehead",
        "Face":          "😐 Face",
    }
    emoji_json = json.dumps(EMOJI)

    html_code = f"""
    <div style="font-family: Georgia, serif; background: #0a0a0f; padding: 10px; border-radius: 16px;">

      <div style="position: relative; display: inline-block; cursor: crosshair;" id="container">

        <img id="faceImg"
             src="data:image/png;base64,{img_b64}"
             style="max-width: 100%; border-radius: 12px; display: block;"
             draggable="false"/>

        <!-- Tooltip -->
        <div id="tooltip" style="
          position: absolute;
          display: none;
          background: rgba(255,255,255,0.92);
          color: #111;
          border: 1px solid rgba(0,0,0,0.12);
          padding: 3px 10px;
          border-radius: 6px;
          font-size: 11px;
          font-weight: 500;
          letter-spacing: 0.3px;
          white-space: nowrap;
          pointer-events: none;
          box-shadow: 0 2px 8px rgba(0,0,0,0.15);
          transform: translate(-50%, -160%);
          z-index: 999;
        "></div>

      </div>

      <!-- Status bar -->
      <div id="statusBar" style="
        margin-top: 12px;
        padding: 10px 20px;
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        color: #aaa;
        font-size: 13px;
        letter-spacing: 1px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.07);
      ">🖱️ Hover over the face to identify features</div>

    </div>

    <script>
      const landmarks = {landmarks_json};
      const emojiMap  = {emoji_json};

      const img       = document.getElementById('faceImg');
      const tooltip   = document.getElementById('tooltip');
      const container = document.getElementById('container');
      const statusBar = document.getElementById('statusBar');

      const natW = {w};
      const natH = {h};
      const THRESHOLD = Math.min(natW, natH) * 0.045;

      function findNearestLabel(ox, oy) {{
        let bestDist  = Infinity;
        let bestLabel = null;
        for (const lm of landmarks) {{
          const dx   = ox - lm.x;
          const dy   = oy - lm.y;
          const dist = Math.sqrt(dx*dx + dy*dy);
          if (dist < bestDist) {{
            bestDist  = dist;
            bestLabel = lm.label;
          }}
        }}
        return bestDist <= THRESHOLD ? bestLabel : null;
      }}

      container.addEventListener('mousemove', function(e) {{
        const rect  = img.getBoundingClientRect();
        const dispX = e.clientX - rect.left;
        const dispY = e.clientY - rect.top;

        const origX = dispX * (natW / rect.width);
        const origY = dispY * (natH / rect.height);

        const label = findNearestLabel(origX, origY);

        if (label) {{
          const display = emojiMap[label] || label;
          tooltip.textContent   = display;
          tooltip.style.display = 'block';
          tooltip.style.left    = dispX + 'px';
          tooltip.style.top     = dispY + 'px';
          statusBar.innerHTML   = '<span style="color:#fff; font-size:15px;">' + display + '</span>';
        }} else {{
          tooltip.style.display = 'none';
          statusBar.innerHTML   = '🖱️ Move over the face features';
        }}
      }});

      container.addEventListener('mouseleave', function() {{
        tooltip.style.display = 'none';
        statusBar.innerHTML   = '🖱️ Hover over the face to identify features';
      }});
    </script>
    """

    display_w = 750
    display_h = int(display_w * (h / w)) + 120
    st.components.v1.html(html_code, height=display_h, scrolling=False)

else:
    st.markdown("""
    <div style="
      border: 2px dashed #2a2a3a;
      border-radius: 16px;
      padding: 60px 40px;
      text-align: center;
      color: #555;
      font-family: Georgia, serif;
      margin-top: 20px;
    ">
      <div style="font-size: 48px; margin-bottom: 16px;">📸</div>
      <div style="font-size: 18px; color: #666;">Upload a front-facing face photo to begin</div>
      <div style="font-size: 13px; color: #444; margin-top: 8px;">JPG · PNG · JPEG supported</div>
    </div>
    """, unsafe_allow_html=True)
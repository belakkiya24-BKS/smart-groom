import streamlit as st
import tensorflow as tf
import easyocr
import cv2
import numpy as np
import pandas as pd
import sqlite3
import re
from PIL import Image
from pathlib import Path
from io import BytesIO
from datetime import datetime

# =========================================================
# 1. APP SETTINGS
# =========================================================

st.set_page_config(
    page_title="✂️ SMART GROOM ✂️",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB = "smart_groom.db"
PHOTO_DIR = Path("student_photos")
PHOTO_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. DATABASE
# =========================================================

def create_database():
    """Create the students table and safely add missing columns."""
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                reg_no TEXT PRIMARY KEY,
                name TEXT,
                course TEXT,
                department TEXT,
                year TEXT,
                before_photo TEXT,
                after_photo TEXT,
                status TEXT DEFAULT 'Pending',
                created_at TEXT
            )
        """)

        # Safe migration for databases created by earlier app versions.
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(students)").fetchall()
        }

        required_columns = {
            "name": "TEXT",
            "course": "TEXT",
            "department": "TEXT",
            "year": "TEXT",
            "before_photo": "TEXT",
            "after_photo": "TEXT",
            "status": "TEXT DEFAULT 'Pending'",
            "created_at": "TEXT"
        }

        for column, column_type in required_columns.items():
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE students ADD COLUMN {column} {column_type}"
                )

        conn.commit()


def get_all_students():
    with sqlite3.connect(DB) as conn:
        return pd.read_sql_query(
            "SELECT * FROM students ORDER BY created_at DESC",
            conn
        )


def get_student(reg_no):
    with sqlite3.connect(DB) as conn:
        result = pd.read_sql_query(
            "SELECT * FROM students WHERE reg_no = ?",
            conn,
            params=(reg_no,)
        )
    return None if result.empty else result.iloc[0]


create_database()


# =========================================================
# 3. OCR ENGINE
# =========================================================
@st.cache_resource
def load_ocr():
    try:
        return easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False
        )
    except Exception as error:
        st.error("EasyOCR Loading Failed!")
        st.exception(error)
        return None


# =========================================================
# 4. GROOMING AI MODEL
# =========================================================

@st.cache_resource
def load_groom_model():
    model_path = Path("groom_model.keras")

    if not model_path.exists():
        return None

    try:
        return tf.keras.models.load_model(model_path)
    except Exception as error:
        st.error(f"AI Model Loading Error: {error}")
        return None


groom_model = load_groom_model()


# =========================================================
# 5. IMAGE PREPROCESSING FOR ID CARD OCR
# =========================================================

def decode_id_image(image_bytes):
    """Read uploaded/camera image and return an RGB numpy image."""
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)


def make_ocr_variants(image_bytes):
    """
    Create multiple OCR-friendly versions of the complete camera image.
    Coordinates from each version are mapped back to the original image.
    """
    rgb = decode_id_image(image_bytes)
    original_height, original_width = rgb.shape[:2]

    # Upscale while preserving aspect ratio.
    target_width = 1800
    scale = target_width / float(original_width)
    target_height = max(1, int(original_height * scale))

    resized = cv2.resize(
        rgb,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # Improve local contrast.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    contrast = clahe.apply(gray)

    # Reduce camera noise.
    denoised = cv2.fastNlMeansDenoising(
        contrast,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Mild sharpening.
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(
        denoised,
        -1,
        sharpen_kernel
    )

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Each variant has the same dimensions, so coordinates are equivalent.
    variants = [
        ("enhanced", sharpened),
        ("adaptive", adaptive),
        ("otsu", otsu)
    ]

    scale_x = original_width / float(target_width)
    scale_y = original_height / float(target_height)

    return variants, sharpened, scale_x, scale_y


def scan_id_card(image_bytes):
    """
    Run EasyOCR on several processed versions.
    Merge near-identical detections and return coordinates in original pixels.
    """
    reader = load_ocr()

    variants, display_image, scale_x, scale_y = make_ocr_variants(
        image_bytes
    )

    detections = []

    for variant_name, variant in variants:
        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            decoder="greedy",
            text_threshold=0.55,
            low_text=0.30,
            link_threshold=0.35
        )

        for box, text, confidence in results:
            text = str(text).strip()

            if not text or float(confidence) < 0.10:
                continue

            center_x = sum(point[0] for point in box) / 4.0
            center_y = sum(point[1] for point in box) / 4.0

            detections.append({
                "text": text,
                "confidence": float(confidence),
                "x": float(center_x * scale_x),
                "y": float(center_y * scale_y),
                "variant": variant_name
            })

    # Merge repeated detections from the three image versions.
    merged = []

    for item in sorted(
        detections,
        key=lambda value: value["confidence"],
        reverse=True
    ):
        normalized = re.sub(
            r"[^A-Z0-9]",
            "",
            item["text"].upper()
        )

        duplicate_index = None

        for index, old in enumerate(merged):
            old_normalized = re.sub(
                r"[^A-Z0-9]",
                "",
                old["text"].upper()
            )

            near_same_position = (
                abs(item["x"] - old["x"]) < 35
                and abs(item["y"] - old["y"]) < 35
            )

            if normalized == old_normalized and near_same_position:
                duplicate_index = index
                break

        if duplicate_index is None:
            merged.append(item)

    merged.sort(key=lambda value: (value["y"], value["x"]))

    return merged, display_image


# =========================================================
# 6. ID CARD LAYOUT-AWARE DETAIL EXTRACTION
# =========================================================

def normalize_ocr_text(value):
    value = value.upper()
    value = value.replace("0", "O")
    value = value.replace("|", "I")
    value = re.sub(r"[^A-Z0-9]", "", value)
    return value


def group_ocr_rows(lines):
    """Group OCR words that appear on approximately the same visual row."""
    if not lines:
        return []

    sorted_lines = sorted(
        lines,
        key=lambda item: (item["y"], item["x"])
    )

    row_tolerance = 22
    rows = []

    for item in sorted_lines:
        placed = False

        for row in rows:
            row_y = float(np.mean([part["y"] for part in row]))

            if abs(item["y"] - row_y) <= row_tolerance:
                row.append(item)
                placed = True
                break

        if not placed:
            rows.append([item])

    output = []

    for row in rows:
        row.sort(key=lambda item: item["x"])

        output.append({
            "text": " ".join(part["text"] for part in row).strip(),
            "x": float(np.mean([part["x"] for part in row])),
            "y": float(np.mean([part["y"] for part in row])),
            "confidence": float(
                np.mean([part["confidence"] for part in row])
            )
        })

    return sorted(output, key=lambda item: item["y"])


def detect_register_number(lines):
    """Find a nine-digit register number, allowing spaces and hyphens."""
    all_text = " ".join(item["text"] for item in lines)

    candidates = re.findall(
        r"(?<!\d)(?:\d[\s-]*){8}\d(?!\d)",
        all_text
    )

    for candidate in candidates:
        digits = re.sub(r"\D", "", candidate)

        if len(digits) == 9:
            return digits

    # OCR sometimes reads a 9-digit number as separate adjacent boxes.
    digit_parts = []

    for item in lines:
        digits = re.sub(r"\D", "", item["text"])
        if digits:
            digit_parts.append((item["y"], item["x"], digits))

    digit_parts.sort(key=lambda item: (item[0], item[1]))

    for _, _, digits in digit_parts:
        if len(digits) == 9:
            return digits

    return ""


def detect_course(rows):
    """
    Recognize the known ID card course.
    This is deliberately format-specific to the sample card supplied.
    """
    for row in rows:
        compact = normalize_ocr_text(row["text"])

        # Common OCR confusions in B.Sc.
        compact = compact.replace("85C", "BSC")
        compact = compact.replace("B5C", "BSC")
        compact = compact.replace("8SC", "BSC")

        has_bsc = "BSC" in compact
        has_computer = "COMPUTER" in compact or "C0MPUTER" in compact
        has_science = "SCIENCE" in compact or "5CIENCE" in compact

        if has_bsc and has_computer and has_science:
            return "B.Sc. - Computer Science", row

    return "", None


def detect_name(rows, course_row):
    """
    In the supplied card layout, the student name is immediately above
    the course. Choose the nearest suitable alphabetic row above it.
    """
    if course_row is None:
        return ""

    candidates = []

    for row in rows:
        if row["y"] >= course_row["y"]:
            continue

        distance = course_row["y"] - row["y"]

        # Name is expected close to the course line on this card.
        if distance > 180:
            continue

        raw = row["text"].strip()

        raw = re.sub(
            r"^(STUDENT\s*NAME|NAME)\s*[:.-]?\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(r"[^A-Za-z .'-]", "", raw).strip()
        letters = re.sub(r"[^A-Za-z]", "", cleaned)
        compact = normalize_ocr_text(cleaned)

        if len(letters) < 3 or len(cleaned) > 40:
            continue

        if len(cleaned.split()) > 5:
            continue

        excluded = (
            "COMPUTER",
            "SCIENCE",
            "COLLEGE",
            "DEPARTMENT",
            "BARCODE",
            "REGISTER",
            "COURSE"
        )

        if any(word in compact for word in excluded):
            continue

        # Prefer the closest row above the course.
        score = distance - (row["confidence"] * 20)
        candidates.append((score, cleaned))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def extract_details(lines):
    rows = group_ocr_rows(lines)

    reg_no = detect_register_number(lines)
    course, course_row = detect_course(rows)
    name = detect_name(rows, course_row)

    department = ""

    if course == "B.Sc. - Computer Science":
        department = "Computer Science"

    return {
        "name": name,
        "reg_no": reg_no,
        "course": course,
        "department": department
    }


# =========================================================
# 7. ACADEMIC YEAR DETECTION
# =========================================================

def detect_year(reg_no, course):
    if not reg_no or len(reg_no) != 9:
        return "Unknown"

    prefix = reg_no[:3]
    course_lower = (course or "").lower()

    # These are the rules supplied in the project brief.
    # Confirm them with the college before final deployment.
    if "b.tech" in course_lower or "btech" in course_lower:
        year_map = {
            "227": "4th Year",
            "228": "3rd Year",
            "229": "2nd Year",
            "230": "1st Year"
            

        }
    else:
        year_map = {
            "227": "3rd Year",
            "228": "2nd Year",
            "229": "1st Year"
        }

    return year_map.get(prefix, "Unknown")


# =========================================================
# 8. GROOMING IMAGE PREDICTION
# =========================================================

def predict_grooming(image_path):
    if groom_model is None:
        return None, None

    try:
        image = Image.open(image_path).convert("RGB")

        input_shape = groom_model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1] or 22
        width = input_shape[2] or 224

        image = image.resize((width, height))

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        prediction = np.asarray(
            groom_model.predict(image_array, verbose=0)
        ).flatten()

        # Expected class mapping:
        # sigmoid: 0 = Groomed, 1 = Not Groomed probability
        # softmax: index 0 = Groomed, index 1 = Not Groomed
        if len(prediction) == 1:
            not_groomed_probability = float(prediction[0])

            if not_groomed_probability >= 0.5:
                result = "Not Groomed"
                confidence = not_groomed_probability
            else:
                result = "Groomed"
                confidence = 1.0 - not_groomed_probability

        elif len(prediction) == 2:
            index = int(np.argmax(prediction))
            result = "Groomed" if index == 0 else "Not Groomed"
            confidence = float(prediction[index])

        else:
            raise ValueError(
                "Model output must contain one sigmoid value or two class values."
            )

        return result, confidence

    except Exception as error:
        st.error(f"Prediction Error: {error}")
        return None, None


# =========================================================
# 9. UI STYLING
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f3f7ff 0%, #e8efff 100%);
    color: #14213d;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071b3a 0%, #123d73 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
h1, h2, h3 {
    color: #102b50 !important;
    font-weight: 750 !important;
}
.stButton > button {
    background: linear-gradient(90deg, #123d73, #2563eb);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 650;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d6e4ff;
    padding: 18px;
    border-radius: 14px;
}
.smart-header {
    background: linear-gradient(100deg, #071b3a, #174b87);
    padding: 25px;
    border-radius: 16px;
    margin-bottom: 25px;
}
.smart-header h1 {
    color: white !important;
    margin-bottom: 5px;
}
.smart-header p {
    color: #dbeafe !important;
    margin: 0;
}
.info-card {
    background: white;
    border-left: 5px solid #2563eb;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 10. HEADER AND NAVIGATION
# =========================================================

st.markdown("""
<div class="smart-header">
    <h1>✂️ SMART GROOM</h1>
    <p>AI-Based Student Grooming Management System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "<h2 style='color:white;'>✂️ SMART GROOM</h2>",
    unsafe_allow_html=True
)
st.sidebar.caption("Student Grooming Management")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "New Registration",
        "Student Search",
        "AI Analysis",
        "Manage Records"
    ]
)


# =========================================================
# 11. DASHBOARD
# =========================================================

if page == "Dashboard":
    st.header("Dashboard")
    st.caption("Student grooming overview")

    data = get_all_students()

    total = len(data)
    pending = int((data["status"] == "Pending").sum()) if total else 0
    groomed = int((data["status"] == "Groomed").sum()) if total else 0
    not_groomed = int((data["status"] == "Not Groomed").sum()) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total)
    c2.metric("Pending", pending)
    c3.metric("Groomed", groomed)
    c4.metric("Not Groomed", not_groomed)

    st.divider()
    st.subheader("Student Records")

    if data.empty:
        st.info("No student records available.")
    else:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )

        excel_buffer = BytesIO()
        data.to_excel(
            excel_buffer,
            index=False,
            engine="openpyxl"
        )
        excel_buffer.seek(0)

        st.download_button(
            "Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name="Smart_Groom_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# =========================================================
# 12. NEW STUDENT REGISTRATION
# =========================================================

elif page == "New Registration":
    st.header("New Student Registration")

    st.info(
        "Step 1: Capture BEFORE grooming photo. "
        "Step 2: Capture the student's ID card."
    )

    before_photo = st.camera_input(
        "Capture Before Grooming Photo",
        key="registration_before"
    )

    id_photo = st.camera_input(
        "Capture Student ID Card",
        key="registration_id"
    )

    if id_photo and st.button("SCAN ID CARD"):
        with st.spinner("Reading ID card. Please wait..."):
            try:
                lines, enhanced = scan_id_card(id_photo.getvalue())
                details = extract_details(lines)

                st.session_state["details"] = details
                st.session_state["ocr_lines"] = lines
                st.session_state["ocr_preview"] = enhanced

            except Exception as error:
                st.error(f"OCR Error: {error}")

    if "details" in st.session_state:
        details = st.session_state["details"]

        st.subheader("Automatically Extracted Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Student Name**")
            st.write(details["name"] or "Not detected")

            st.write("**Course**")
            st.write(details["course"] or "Not detected")

        with col2:
            st.write("**Register Number**")
            st.write(details["reg_no"] or "Not detected")

            st.write("**Department**")
            st.write(details["department"] or "Not detected")

        year = detect_year(
            details["reg_no"],
            details["course"]
        )
        st.info(f"Academic Year: {year}")

        if "ocr_preview" in st.session_state:
            with st.expander("View Enhanced ID Card Image"):
                st.image(
                    st.session_state["ocr_preview"],
                    caption="Enhanced grayscale image",
                    clamp=True
                )

        with st.expander("View OCR Text and Confidence"):
            for item in st.session_state.get("ocr_lines", []):
                st.write(
                    f"{item['text']} — Confidence: "
                    f"{item['confidence']:.2f}"
                )

        if st.button("SAVE STUDENT"):
            missing = []

            if not details["name"]:
                missing.append("Student Name")
            if not details["reg_no"]:
                missing.append("Register Number")
            if not details["course"]:
                missing.append("Course")
            if not details["department"]:
                missing.append("Department")
            if not before_photo:
                missing.append("BEFORE Photo")
            if year == "Unknown":
                missing.append("Verified Academic Year")

            if missing:
                st.error(
                    "Cannot save. Required information not detected: "
                    + ", ".join(missing)
                    + ". Please capture a clearer ID card image."
                )
            else:
                reg_no = details["reg_no"]
                photo_path = PHOTO_DIR / f"{reg_no}_before.jpg"

                try:
                    with sqlite3.connect(DB) as conn:
                        exists = conn.execute(
                            "SELECT 1 FROM students WHERE reg_no = ?",
                            (reg_no,)
                        ).fetchone()

                        if exists:
                            st.error(
                                "This Register Number already exists."
                            )
                        else:
                            photo_path.write_bytes(
                                before_photo.getvalue()
                            )

                            conn.execute("""
                                INSERT INTO students (
                                    reg_no, name, course, department, year,
                                    before_photo, after_photo, status, created_at
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                reg_no,
                                details["name"],
                                details["course"],
                                details["department"],
                                year,
                                str(photo_path),
                                "",
                                "Pending",
                                datetime.now().strftime("%Y-%m-%d %H:%M")
                            ))

                            conn.commit()

                            st.success("Student Registered Successfully!")

                            for key in [
                                "details",
                                "ocr_lines",
                                "ocr_preview"
                            ]:
                                st.session_state.pop(key, None)

                            st.rerun()

                except sqlite3.Error as error:
                    st.error(f"Database Error: {error}")


# =========================================================
# 13. STUDENT SEARCH AND AFTER PHOTO
# =========================================================

elif page == "Student Search":
    st.header("Student Search")
    st.write("Search using the student's Register Number.")

    search_reg = st.text_input(
        "Enter Register Number",
        max_chars=9
    )

    if st.button("SEARCH STUDENT"):
        student = get_student(search_reg.strip())

        if student is not None:
            st.session_state["selected_reg"] = search_reg.strip()
            st.success("Student Found!")
        else:
            st.session_state.pop("selected_reg", None)
            st.warning("Student Not Found.")

    selected_reg = st.session_state.get("selected_reg")

    if selected_reg:
        student = get_student(selected_reg)

        if student is not None:
            st.divider()
            st.subheader("Student Details")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Name:** {student['name']}")
                st.write(f"**Register Number:** {student['reg_no']}")
                st.write(f"**Course:** {student['course']}")

            with col2:
                st.write(f"**Department:** {student['department']}")
                st.write(f"**Academic Year:** {student['year']}")
                st.write(f"**Status:** {student['status']}")

            before_path = Path(student["before_photo"]) if student["before_photo"] else None

            if before_path and before_path.exists():
                st.image(
                    str(before_path),
                    caption="Before Grooming Photo",
                    width=350
                )
            else:
                st.warning("Before photo file not found.")

            st.divider()
            st.subheader("Capture AFTER Grooming Photo")

            after_photo = st.camera_input(
                "Capture After Photo",
                key=f"after_{selected_reg}"
            )

            if after_photo and st.button("SAVE AFTER PHOTO"):
                after_path = PHOTO_DIR / f"{selected_reg}_after.jpg"

                try:
                    after_path.write_bytes(after_photo.getvalue())

                    with sqlite3.connect(DB) as conn:
                        conn.execute(
                            """
                            UPDATE students
                            SET after_photo = ?
                            WHERE reg_no = ?
                            """,
                            (str(after_path), selected_reg)
                        )
                        conn.commit()

                    st.success(
                        "After Photo Saved. Go to AI Analysis."
                    )
                    st.rerun()

                except (OSError, sqlite3.Error) as error:
                    st.error(f"Could not save AFTER photo: {error}")


# =========================================================
# 14. AI ANALYSIS AND TEACHER CONFIRMATION
# =========================================================

elif page == "AI Analysis":
    st.header("AI Grooming Analysis")

    data = get_all_students()

    if data.empty:
        st.info("No student records available.")
    else:
        available = data[
            data["after_photo"].fillna("") != ""
        ]

        if available.empty:
            st.warning(
                "No AFTER photos available. Capture an AFTER photo "
                "from Student Search first."
            )
        else:
            options = {
                f"{row['name']} - {row['reg_no']}": row["reg_no"]
                for _, row in available.iterrows()
            }

            selected_label = st.selectbox(
                "Select Student",
                list(options.keys())
            )

            selected_reg = options[selected_label]
            student = get_student(selected_reg)

            before_path = Path(student["before_photo"]) if student["before_photo"] else None
            after_path = Path(student["after_photo"])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Before Grooming")
                if before_path and before_path.exists():
                    st.image(str(before_path), use_container_width=True)
                else:
                    st.warning("Before photo not found.")

            with col2:
                st.subheader("After Grooming")
                if after_path.exists():
                    st.image(str(after_path), use_container_width=True)
                else:
                    st.warning("After photo not found.")

            st.caption(
                "AI output is only a suggestion. The teacher must review "
                "the photographs and make the final decision."
            )

            if groom_model is None:
                st.error(
                    "AI model is not available. Check that "
                    "groom_model.keras exists and can be loaded."
                )
            else:
                if st.button("RUN AI ANALYSIS"):
                    with st.spinner("AI is analysing the AFTER photo..."):
                        result, confidence = predict_grooming(str(after_path))

                    if result is not None:
                        st.session_state["ai_result"] = result
                        st.session_state["ai_confidence"] = confidence
                        st.session_state["ai_reg"] = selected_reg

                if st.session_state.get("ai_reg") == selected_reg:
                    st.divider()
                    st.subheader("AI Suggestion")

                    st.info(
                        f"AI Suggestion: {st.session_state['ai_result']}"
                    )
                    st.write(
                        "Prediction confidence: "
                        f"{st.session_state['ai_confidence'] * 100:.2f}%"
                    )

                    st.subheader("Teacher Final Decision")

                    final_decision = st.radio(
                        "Review both photos and select the final status:",
                        ["Groomed", "Not Groomed"],
                        horizontal=True,
                        key=f"final_decision_{selected_reg}"
                    )

                    if st.button("CONFIRM FINAL DECISION"):
                        try:
                            with sqlite3.connect(DB) as conn:
                                conn.execute(
                                    """
                                    UPDATE students
                                    SET status = ?
                                    WHERE reg_no = ?
                                    """,
                                    (final_decision, selected_reg)
                                )
                                conn.commit()

                            st.success(
                                f"Teacher-confirmed status saved: {final_decision}"
                            )

                            for key in [
                                "ai_reg",
                                "ai_result",
                                "ai_confidence"
                            ]:
                                st.session_state.pop(key, None)

                            st.rerun()

                        except sqlite3.Error as error:
                            st.error(f"Database update failed: {error}")


# =========================================================
# 15. RECORD MANAGEMENT
# =========================================================

elif page == "Manage Records":
    st.header("Manage Student Records")

    data = get_all_students()

    if data.empty:
        st.info("No student records available.")
    else:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )

        selected_reg = st.selectbox(
            "Select Register Number to delete",
            data["reg_no"].tolist()
        )

        st.warning(
            "Deleting a record cannot be undone. "
            "The associated photo files will also be deleted."
        )

        confirm_delete = st.checkbox(
            "I understand and want to delete this student record."
        )

        if st.button(
            "DELETE STUDENT RECORD",
            disabled=not confirm_delete
        ):
            student = get_student(selected_reg)

            try:
                with sqlite3.connect(DB) as conn:
                    conn.execute(
                        "DELETE FROM students WHERE reg_no = ?",
                        (selected_reg,)
                    )
                    conn.commit()

                if student is not None:
                    for photo_column in ["before_photo", "after_photo"]:
                        photo_value = student[photo_column]

                        if photo_value:
                            photo_path = Path(photo_value)

                            # Only delete files inside the project photo folder.
                            try:
                                photo_path.resolve().relative_to(
                                    PHOTO_DIR.resolve()
                                )
                                if photo_path.exists():
                                    photo_path.unlink()
                            except (ValueError, OSError):
                                pass

                st.success("Student record deleted.")
                st.session_state.pop("selected_reg", None)
                st.rerun()

            except sqlite3.Error as error:
                st.error(f"Delete failed: {error}")

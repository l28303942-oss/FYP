import os
import re
import uuid
from datetime import date, datetime

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import or_
from werkzeug.utils import secure_filename

from app.db_models import Patient, Report, ScanRecord, User
from app.extensions import db
from app.services.pdf_generator import generate_scan_report
from ml.inference_service import run_inference, save_visualizations

main_bp = Blueprint("main", __name__)


def _allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


def _next_patient_id():
    prefix = f"P-{date.today().year}"
    last = (
        Patient.query.filter(Patient.patient_id.like(f"{prefix}%"))
        .order_by(Patient.id.desc())
        .first()
    )
    n = 1
    if last and last.patient_id:
        m = re.search(r"(\d+)$", last.patient_id)
        if m:
            n = int(m.group(1)) + 1
    return f"{prefix}{n:04d}"


@main_bp.route("/dashboard")
@login_required
def dashboard():
    total_patients = Patient.query.filter_by(user_id=current_user.id).count()
    total_scans = ScanRecord.query.filter_by(user_id=current_user.id).count()
    segmented = (
        ScanRecord.query.filter_by(user_id=current_user.id)
        .filter(ScanRecord.status == "completed")
        .count()
    )
    pending = (
        ScanRecord.query.filter_by(user_id=current_user.id)
        .filter(ScanRecord.status.in_(["uploaded", "processing"]))
        .count()
    )
    acc = 0.0
    done = (
        ScanRecord.query.filter_by(user_id=current_user.id)
        .filter(ScanRecord.status == "completed", ScanRecord.confidence.isnot(None))
        .all()
    )
    if done:
        acc = sum(s.confidence or 0 for s in done) / len(done) * 100

    recent = (
        ScanRecord.query.filter_by(user_id=current_user.id)
        .order_by(ScanRecord.created_at.desc())
        .limit(8)
        .all()
    )
    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_scans=total_scans,
        segmented=segmented,
        pending=pending,
        accuracy=round(acc, 1),
        recent=recent,
    )


@main_bp.route("/api/stats")
@login_required
def api_stats():
    return jsonify(
        {
            "patients": Patient.query.filter_by(user_id=current_user.id).count(),
            "scans": ScanRecord.query.filter_by(user_id=current_user.id).count(),
        }
    )


@main_bp.route("/patients", methods=["GET", "POST"])
@login_required
def patients():
    q = (request.args.get("q") or "").strip()
    gender = (request.args.get("gender") or "").strip()
    query = Patient.query.filter_by(user_id=current_user.id)
    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                Patient.full_name.ilike(like),
                Patient.patient_id.ilike(like),
                Patient.contact.ilike(like),
            )
        )
    if gender in ("Male", "Female", "Other"):
        query = query.filter_by(gender=gender)
    rows = query.order_by(Patient.created_at.desc()).all()
    return render_template("patients.html", patients=rows, q=q, gender=gender)


@main_bp.route("/patients/add", methods=["POST"])
@login_required
def patient_add():
    try:
        scan_date = request.form.get("scan_date") or None
        sd = datetime.strptime(scan_date, "%Y-%m-%d").date() if scan_date else None
        p = Patient(
            patient_id=_next_patient_id(),
            full_name=request.form.get("full_name", "").strip(),
            age=int(request.form.get("age") or 0),
            gender=request.form.get("gender", "Other"),
            contact=request.form.get("contact", "").strip(),
            disease_notes=request.form.get("disease_notes", "").strip(),
            scan_date=sd,
            doctor_name=request.form.get("doctor_name", "").strip(),
            user_id=current_user.id,
        )
        db.session.add(p)
        db.session.commit()
        flash("Patient added successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Could not add patient: {e}", "danger")
    return redirect(url_for("main.patients"))


@main_bp.route("/patients/<int:pid>/edit", methods=["POST"])
@login_required
def patient_edit(pid):
    p = Patient.query.filter_by(id=pid, user_id=current_user.id).first_or_404()
    try:
        p.full_name = request.form.get("full_name", p.full_name).strip()
        p.age = int(request.form.get("age") or p.age)
        p.gender = request.form.get("gender", p.gender)
        p.contact = request.form.get("contact", "").strip()
        p.disease_notes = request.form.get("disease_notes", "").strip()
        sd = request.form.get("scan_date")
        p.scan_date = datetime.strptime(sd, "%Y-%m-%d").date() if sd else None
        p.doctor_name = request.form.get("doctor_name", "").strip()
        db.session.commit()
        flash("Patient updated.", "success")
    except Exception as e:
        db.session.rollback()
        flash(str(e), "danger")
    return redirect(url_for("main.patients"))


@main_bp.route("/patients/<int:pid>/delete", methods=["POST"])
@login_required
def patient_delete(pid):
    p = Patient.query.filter_by(id=pid, user_id=current_user.id).first_or_404()
    db.session.delete(p)
    db.session.commit()
    flash("Patient removed.", "info")
    return redirect(url_for("main.patients"))


@main_bp.route("/media/scan/<int:sid>")
@login_required
def media_scan(sid):
    rec = ScanRecord.query.filter_by(id=sid, user_id=current_user.id).first_or_404()
    if not rec.file_path or not os.path.isfile(rec.file_path):
        abort(404)
    return send_file(rec.file_path)


@main_bp.route("/media/overlay/<int:sid>")
@login_required
def media_overlay(sid):
    rec = ScanRecord.query.filter_by(id=sid, user_id=current_user.id).first_or_404()
    if not rec.overlay_path or not os.path.isfile(rec.overlay_path):
        abort(404)
    return send_file(rec.overlay_path)


@main_bp.route("/media/heatmap/<int:sid>")
@login_required
def media_heatmap(sid):
    rec = ScanRecord.query.filter_by(id=sid, user_id=current_user.id).first_or_404()
    if not rec.heatmap_path or not os.path.isfile(rec.heatmap_path):
        abort(404)
    return send_file(rec.heatmap_path)


@main_bp.route("/media/mask/<int:sid>")
@login_required
def media_mask(sid):
    rec = ScanRecord.query.filter_by(id=sid, user_id=current_user.id).first_or_404()
    if not rec.mask_path or not os.path.isfile(rec.mask_path):
        abort(404)
    return send_file(rec.mask_path)


@main_bp.route("/api/scan/upload", methods=["POST"])
@login_required
def api_scan_upload():
    """XHR/json upload for progress bar support."""
    pid = request.form.get("patient_id")
    file = request.files.get("scan_file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "No file"}), 400
    if not _allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Invalid file type"}), 400
    patient = Patient.query.filter_by(id=int(pid or 0), user_id=current_user.id).first()
    if not patient:
        return jsonify({"ok": False, "error": "Invalid patient"}), 400

    raw_name = secure_filename(file.filename)
    uid = uuid.uuid4().hex[:12]
    ext = raw_name.rsplit(".", 1)[1].lower()
    fname = f"{patient.patient_id}_{uid}.{ext}"
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], fname)
    file.save(path)

    rec = ScanRecord(
        patient_id=patient.id,
        user_id=current_user.id,
        original_filename=raw_name,
        file_path=path,
        file_type=ext,
        status="uploaded",
    )
    db.session.add(rec)
    db.session.commit()
    return jsonify({"ok": True, "scan_id": rec.id, "message": "Uploaded"})


@main_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    patient_list = Patient.query.filter_by(user_id=current_user.id).order_by(Patient.full_name).all()
    if request.method == "POST":
        pid = request.form.get("patient_id")
        file = request.files.get("scan_file")
        if not file or not file.filename:
            flash("Please choose a file.", "warning")
            return redirect(url_for("main.upload"))
        if not _allowed_file(file.filename):
            flash("Allowed: PNG, JPG, JPEG, DICOM (.dcm)", "danger")
            return redirect(url_for("main.upload"))
        patient = Patient.query.filter_by(id=int(pid), user_id=current_user.id).first()
        if not patient:
            flash("Invalid patient.", "danger")
            return redirect(url_for("main.upload"))

        raw_name = secure_filename(file.filename)
        uid = uuid.uuid4().hex[:12]
        ext = raw_name.rsplit(".", 1)[1].lower()
        fname = f"{patient.patient_id}_{uid}.{ext}"
        path = os.path.join(current_app.config["UPLOAD_FOLDER"], fname)
        file.save(path)

        rec = ScanRecord(
            patient_id=patient.id,
            user_id=current_user.id,
            original_filename=raw_name,
            file_path=path,
            file_type=ext,
            status="uploaded",
        )
        db.session.add(rec)
        db.session.commit()
        flash("Scan uploaded. Run segmentation from the Segmentation page.", "success")
        return redirect(url_for("main.segmentation"))

    return render_template("upload.html", patients=patient_list)


@main_bp.route("/segmentation", methods=["GET", "POST"])
@login_required
def segmentation():
    scans = (
        ScanRecord.query.filter_by(user_id=current_user.id)
        .order_by(ScanRecord.created_at.desc())
        .all()
    )
    patients = {p.id: p for p in Patient.query.filter_by(user_id=current_user.id).all()}

    if request.method == "POST":
        scan_id = int(request.form.get("scan_id") or 0)
        rec = ScanRecord.query.filter_by(id=scan_id, user_id=current_user.id).first_or_404()
        rec.status = "processing"
        db.session.commit()

        weights = current_app.config.get("MODEL_WEIGHTS")
        try:
            result = run_inference(rec.file_path, weights_path=weights)
            base = uuid.uuid4().hex[:10]
            out_dir = current_app.config["RESULT_FOLDER"]
            mask_p, heat_p, over_p = save_visualizations(
                rec.file_path, result.mask_array, out_dir, base
            )
            rec.mask_path = mask_p
            rec.heatmap_path = heat_p
            rec.overlay_path = over_p
            rec.confidence = result.confidence
            rec.processing_time_sec = result.processing_time_sec
            rec.status = "completed"
            rec.error_message = result.message
            db.session.commit()
            flash(result.message, "success" if result.used_real_model else "warning")
        except Exception as e:
            rec.status = "failed"
            rec.error_message = str(e)
            db.session.commit()
            flash(f"Segmentation failed: {e}", "danger")

        return redirect(url_for("main.segmentation"))

    return render_template(
        "segmentation.html",
        scans=scans,
        patients=patients,
    )


@main_bp.route("/reports", methods=["GET", "POST"])
@login_required
def reports():
    scans = (
        ScanRecord.query.filter_by(user_id=current_user.id)
        .filter(ScanRecord.status == "completed")
        .order_by(ScanRecord.created_at.desc())
        .all()
    )
    patients = {p.id: p for p in Patient.query.filter_by(user_id=current_user.id).all()}
    existing = {r.scan_id: r for r in Report.query.filter_by(user_id=current_user.id).all()}

    if request.method == "POST":
        scan_id = int(request.form.get("scan_id") or 0)
        rec = ScanRecord.query.filter_by(id=scan_id, user_id=current_user.id).first_or_404()
        if rec.status != "completed":
            flash("Complete segmentation first.", "warning")
            return redirect(url_for("main.reports"))
        pat = Patient.query.get(rec.patient_id)
        pdf_name = f"report_{scan_id}_{uuid.uuid4().hex[:8]}.pdf"
        pdf_path = os.path.join(current_app.config["BASE_DIR"], "uploads", "reports", pdf_name)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        generate_scan_report(
            hospital_name=current_app.config["HOSPITAL_NAME"],
            patient_name=pat.full_name,
            patient_id=pat.patient_id,
            scan_date=pat.scan_date.isoformat() if pat.scan_date else "—",
            doctor_name=pat.doctor_name or "",
            confidence=float(rec.confidence or 0),
            processing_time=float(rec.processing_time_sec or 0),
            original_image_path=rec.file_path if rec.file_type in ("png", "jpg", "jpeg") else None,
            overlay_image_path=rec.overlay_path,
            out_pdf_path=pdf_path,
        )

        rep = Report(scan_id=rec.id, user_id=current_user.id, pdf_path=pdf_path)
        db.session.add(rep)
        db.session.commit()
        flash("PDF report generated.", "success")
        return redirect(url_for("main.reports"))

    return render_template(
        "reports.html",
        scans=scans,
        patients=patients,
        existing=existing,
    )


@main_bp.route("/reports/download/<int:rid>")
@login_required
def download_report(rid):
    r = Report.query.filter_by(id=rid, user_id=current_user.id).first_or_404()
    if not r.pdf_path or not os.path.isfile(r.pdf_path):
        flash("File missing.", "danger")
        return redirect(url_for("main.reports"))
    return send_file(r.pdf_path, as_attachment=True, download_name=os.path.basename(r.pdf_path))


@main_bp.route("/history")
@login_required
def history():
    q = (request.args.get("q") or "").strip()
    sort = request.args.get("sort") or "desc"
    patient_filter = request.args.get("patient_id")

    query = ScanRecord.query.filter_by(user_id=current_user.id)
    if patient_filter:
        query = query.filter_by(patient_id=int(patient_filter))
    if q:
        # join patient name search
        query = query.join(Patient).filter(Patient.full_name.ilike(f"%{q}%"))

    if sort == "asc":
        query = query.order_by(ScanRecord.created_at.asc())
    else:
        query = query.order_by(ScanRecord.created_at.desc())

    rows = query.all()
    patients = Patient.query.filter_by(user_id=current_user.id).order_by(Patient.full_name).all()
    reps = {r.scan_id: r for r in Report.query.filter_by(user_id=current_user.id).all()}
    return render_template(
        "history.html",
        scans=rows,
        patients_map={p.id: p for p in patients},
        patient_list=patients,
        reps=reps,
        q=q,
        sort=sort,
        patient_filter=patient_filter or "",
    )


@main_bp.route("/history/delete/<int:sid>", methods=["POST"])
@login_required
def history_delete(sid):
    rec = ScanRecord.query.filter_by(id=sid, user_id=current_user.id).first_or_404()
    Report.query.filter_by(scan_id=rec.id).delete()
    db.session.delete(rec)
    db.session.commit()
    flash("Record removed.", "info")
    return redirect(url_for("main.history"))


@main_bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    user = User.query.get(current_user.id)
    if request.method == "POST":
        action = request.form.get("action")
        if action == "profile":
            user.email = (request.form.get("email") or user.email).strip().lower()
            db.session.commit()
            flash("Profile updated.", "success")
        elif action == "password":
            cur = request.form.get("current_password") or ""
            new = request.form.get("new_password") or ""
            conf = request.form.get("confirm_password") or ""
            if not user.check_password(cur):
                flash("Current password incorrect.", "danger")
            elif len(new) < 8:
                flash("New password too short.", "danger")
            elif new != conf:
                flash("Passwords do not match.", "danger")
            else:
                user.set_password(new)
                db.session.commit()
                flash("Password changed.", "success")
        elif action == "prefs":
            user.theme = request.form.get("theme") or user.theme
            user.notifications_enabled = bool(request.form.get("notifications"))
            db.session.commit()
            flash("Preferences saved.", "success")
        return redirect(url_for("main.settings"))

    return render_template("settings.html", user=user)

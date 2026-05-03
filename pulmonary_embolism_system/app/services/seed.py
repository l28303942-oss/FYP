"""Insert demo users and patients when DB is empty."""
from datetime import date, timedelta

from app.db_models import Patient, User
from app.extensions import db


def seed_if_empty():
    if User.query.first():
        return
    u = User(username="demo", email="demo@hospital.edu")
    u.set_password("demo12345")
    db.session.add(u)
    db.session.commit()

    p1 = Patient(
        patient_id="P-2026001",
        full_name="John Carter",
        age=54,
        gender="Male",
        contact="+1-555-0101",
        disease_notes="Suspected PE — contrast CT scheduled.",
        scan_date=date.today() - timedelta(days=2),
        doctor_name="Dr. Sarah Khan",
        user_id=u.id,
    )
    p2 = Patient(
        patient_id="P-2026002",
        full_name="Emily Chen",
        age=41,
        gender="Female",
        contact="+1-555-0199",
        disease_notes="Follow-up after anticoagulation.",
        scan_date=date.today() - timedelta(days=5),
        doctor_name="Dr. Michael Ross",
        user_id=u.id,
    )
    db.session.add_all([p1, p2])
    db.session.commit()

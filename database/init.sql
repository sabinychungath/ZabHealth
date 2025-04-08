CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    medical_history TEXT
);

CREATE TABLE consultations (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    transcript TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

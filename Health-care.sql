-- ============================================
-- DATABASE INITIALIZATION
-- ============================================
IF DB_ID('HealthAI_Project') IS NOT NULL
    DROP DATABASE HealthAI_Project;
GO

CREATE DATABASE HealthAI_Project;
GO

USE HealthAI_Project;
GO

-- ============================================
-- 1. GEOGRAPHIC REGIONS
-- ============================================
CREATE TABLE Geographic_Regions (
    RegionID INT PRIMARY KEY IDENTITY(1,1),
    RegionName VARCHAR(100) NOT NULL,
    SocioEconomicIndex FLOAT CHECK (SocioEconomicIndex BETWEEN 0 AND 1),
    PollutionLevel FLOAT CHECK (PollutionLevel >= 0)
);
GO

-- ============================================
-- 2. PATIENTS
-- ============================================
CREATE TABLE Patients (
    PatientID INT PRIMARY KEY IDENTITY(1,1),
    Age INT NOT NULL CHECK (Age BETWEEN 0 AND 120),
    Gender VARCHAR(10) NOT NULL CHECK (Gender IN ('Male', 'Female', 'Other')),
    RegionID INT NOT NULL,
    Created_At DATETIME DEFAULT GETDATE(),

    CONSTRAINT FK_Patients_Region
        FOREIGN KEY (RegionID) REFERENCES Geographic_Regions(RegionID)
);
GO

-- ============================================
-- 3. BLOOD PRESSURE LOGS
-- ============================================
CREATE TABLE BloodPressure_Logs (
    BPLogID INT PRIMARY KEY IDENTITY(1,1),
    PatientID INT NOT NULL,
    ReadingDate DATE NOT NULL,
    Systolic INT NOT NULL CHECK (Systolic BETWEEN 50 AND 250),
    Diastolic INT NOT NULL CHECK (Diastolic BETWEEN 30 AND 150),
    Pulse INT CHECK (Pulse BETWEEN 30 AND 200),

    CONSTRAINT FK_BPLogs_Patient
        FOREIGN KEY (PatientID) REFERENCES Patients(PatientID)
);
GO

-- ============================================
-- 4. BIOMETRIC HISTORY
-- ============================================
CREATE TABLE Biometric_History (
    BioID INT PRIMARY KEY IDENTITY(1,1),
    PatientID INT NOT NULL,
    RecordDate DATE NOT NULL,
    Weight FLOAT CHECK (Weight > 0),
    BMI FLOAT CHECK (BMI > 0),

    CONSTRAINT FK_Bio_Patient
        FOREIGN KEY (PatientID) REFERENCES Patients(PatientID)
);
GO

-- ============================================
-- 5. AGGREGATED STATS (FEATURE ENGINEERING)
-- ============================================
CREATE TABLE Aggregated_Stats (
    StatID INT PRIMARY KEY IDENTITY(1,1),
    PatientID INT NOT NULL UNIQUE,

    Avg_Systolic FLOAT CHECK (Avg_Systolic BETWEEN 50 AND 250),
    Avg_Diastolic FLOAT CHECK (Avg_Diastolic BETWEEN 30 AND 150),
    BP_Volatility FLOAT CHECK (BP_Volatility >= 0),
    Pulse_Pressure FLOAT CHECK (Pulse_Pressure >= 0),
    Reading_Count INT NOT NULL CHECK (Reading_Count >= 0),

    Last_Updated DATETIME DEFAULT GETDATE(),

    CONSTRAINT FK_Stats_Patient
        FOREIGN KEY (PatientID) REFERENCES Patients(PatientID)
);
GO

-- ============================================
-- 6. CLINICAL EVENTS
-- ============================================
CREATE TABLE Clinical_Events (
    EventID INT PRIMARY KEY IDENTITY(1,1),
    PatientID INT NOT NULL,
    EventDate DATE NOT NULL,
    EventType VARCHAR(50) NOT NULL CHECK (
        EventType IN ('Hypertension', 'Stroke', 'Heart Attack')
    ),

    CONSTRAINT FK_Events_Patient
        FOREIGN KEY (PatientID) REFERENCES Patients(PatientID)
);
GO

-- ============================================
-- 7. MEDICATION RECORDS
-- ============================================
CREATE TABLE Medication_Records (
    MedicationID INT PRIMARY KEY IDENTITY(1,1),
    PatientID INT NOT NULL,
    MedicationName VARCHAR(100) NOT NULL,
    Dosage VARCHAR(50),
    StartDate DATE,
    EndDate DATE,

    CONSTRAINT FK_Medication_Patient
        FOREIGN KEY (PatientID) REFERENCES Patients(PatientID)
);
GO

-- ============================================
-- 8. EMERGENCY LOGS
-- ============================================
CREATE TABLE Emergency_Logs (
    EmergencyID INT PRIMARY KEY IDENTITY(1,1),
    PatientID INT NOT NULL,
    VisitDate DATE NOT NULL,
    Reason VARCHAR(100),
    Severity VARCHAR(20) CHECK (Severity IN ('Low', 'Medium', 'High')),

    CONSTRAINT FK_Emergency_Patient
        FOREIGN KEY (PatientID) REFERENCES Patients(PatientID)
);
GO

-- ============================================
-- 9. MODEL REGISTRY
-- ============================================
CREATE TABLE Model_Registry (
    ModelID INT PRIMARY KEY IDENTITY(1,1),
    ModelName VARCHAR(100) NOT NULL,
    ModelType VARCHAR(50) NOT NULL CHECK (
        ModelType IN ('LogisticRegression', 'RandomForestClassifier', 'RandomForestRegressor')
    ),
    Version VARCHAR(20) NOT NULL,
    TrainingDate DATETIME DEFAULT GETDATE(),
    Accuracy FLOAT CHECK (Accuracy BETWEEN 0 AND 1),
    RMSE FLOAT CHECK (RMSE >= 0)
);
GO

-- ============================================
-- 10. PREDICTION RESULTS
-- ============================================
CREATE TABLE Prediction_Results (
    PredictionID INT PRIMARY KEY IDENTITY(1,1),
    PatientID INT NOT NULL,
    ModelID INT NOT NULL,

    PredictionDate DATETIME DEFAULT GETDATE(),

    Risk_Score FLOAT CHECK (Risk_Score BETWEEN 0 AND 1),
    Risk_Tier VARCHAR(20) CHECK (Risk_Tier IN ('Low', 'Medium', 'High')),
    Predicted_Value FLOAT,
    Probability FLOAT CHECK (Probability BETWEEN 0 AND 1),

    CONSTRAINT FK_Prediction_Patient
        FOREIGN KEY (PatientID) REFERENCES Patients(PatientID),

    CONSTRAINT FK_Prediction_Model
        FOREIGN KEY (ModelID) REFERENCES Model_Registry(ModelID)
);
GO

-- ============================================
-- 11. RESEARCH ASSESSMENTS (dashboard audit trail)
-- ============================================
CREATE TABLE Research_Assessments (
    AssessmentID INT PRIMARY KEY IDENTITY(1,1),
    PatientID INT NULL,
    ResearcherName VARCHAR(100) NULL,
    InputSource VARCHAR(20) NOT NULL CHECK (
        InputSource IN ('Manual', 'Database', 'Demo')
    ),

    Age INT NOT NULL CHECK (Age BETWEEN 0 AND 120),
    Gender VARCHAR(10) NOT NULL,
    RegionID INT NULL,
    Avg_Systolic FLOAT NOT NULL,
    Avg_Diastolic FLOAT NOT NULL,
    BP_Volatility FLOAT NOT NULL,
    Pulse_Pressure FLOAT NOT NULL,
    Reading_Count INT NOT NULL,

    Overall_Risk_Level VARCHAR(20) NOT NULL CHECK (
        Overall_Risk_Level IN ('Low', 'Moderate', 'High', 'Critical')
    ),
    Overall_Risk_Score FLOAT NOT NULL,
    Overall_Risk_Flag INT NOT NULL CHECK (Overall_Risk_Flag IN (0, 1)),
    Pipeline_Version VARCHAR(20) NOT NULL DEFAULT 'v2',
    Assessed_At DATETIME NOT NULL DEFAULT GETDATE(),
    Notes VARCHAR(500) NULL,

    CONSTRAINT FK_Research_Patient
        FOREIGN KEY (PatientID) REFERENCES Patients(PatientID)
);
GO

-- ============================================
-- 12. RESEARCH PIPELINE STEPS (per-stage results)
-- ============================================
CREATE TABLE Research_Pipeline_Steps (
    StepID INT PRIMARY KEY IDENTITY(1,1),
    AssessmentID INT NOT NULL,
    StepOrder INT NOT NULL,
    StageName VARCHAR(100) NOT NULL,
    StageType VARCHAR(20) NOT NULL,
    Predicted_Value FLOAT NOT NULL,
    Depends_On_JSON VARCHAR(MAX) NULL,

    CONSTRAINT FK_Research_Step_Assessment
        FOREIGN KEY (AssessmentID) REFERENCES Research_Assessments(AssessmentID)
        ON DELETE CASCADE
);
GO

-- ============================================
-- INDEXES FOR PERFORMANCE (IMPORTANT)
-- ============================================

CREATE INDEX IDX_BPLogs_Patient ON BloodPressure_Logs(PatientID);
CREATE INDEX IDX_Bio_Patient ON Biometric_History(PatientID);
CREATE INDEX IDX_Stats_Patient ON Aggregated_Stats(PatientID);
CREATE INDEX IDX_Events_Patient ON Clinical_Events(PatientID);
CREATE INDEX IDX_Emergency_Patient ON Emergency_Logs(PatientID);
CREATE INDEX IDX_Research_AssessedAt ON Research_Assessments(Assessed_At DESC);
CREATE INDEX IDX_Research_Patient ON Research_Assessments(PatientID);
CREATE INDEX IDX_Research_Steps_Assessment ON Research_Pipeline_Steps(AssessmentID);
GO


-- Run this in SSMS on an EXISTING HealthAI_Project database
-- (does not drop or recreate the database)

USE HealthAI_Project;
GO

IF OBJECT_ID('Research_Pipeline_Steps', 'U') IS NOT NULL
    DROP TABLE Research_Pipeline_Steps;
GO

IF OBJECT_ID('Research_Assessments', 'U') IS NOT NULL
    DROP TABLE Research_Assessments;
GO

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

CREATE INDEX IDX_Research_AssessedAt ON Research_Assessments(Assessed_At DESC);
CREATE INDEX IDX_Research_Patient ON Research_Assessments(PatientID);
CREATE INDEX IDX_Research_Steps_Assessment ON Research_Pipeline_Steps(AssessmentID);
GO

PRINT 'Research tracking tables created successfully.';
GO

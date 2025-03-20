![poster](https://github.com/user-attachments/assets/e863dd46-6f16-4a6f-a813-3005decfd85d)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Project Demo and Documents](#project-demo-and-documents)
- [Project Architecture](#project-architecture)
- [User Interface](#user-interface)
- [Future Work](#future-work)

---

## Introduction
Face Liveness Detection System is a biometric security project aimed at preventing spoofing attacks using photos, videos, and masks. It incorporates three techniques:
1. **Eyeball Movement Analysis**
2. **Texture Analysis**
3. **Blood Flow Analysis**

The system enhances digital security by providing real-time verification of user presence.

## Features
- Real-time face detection and liveness verification.
- Supports multiple anti-spoofing techniques.
- User-friendly interface with multiple pages.
- Secure and privacy-focused implementation.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Backend**: Django (Python)
- **Machine Learning**: OpenCV, Scikit-learn, Mediapipe, NumPy
- **Data Processing**: PIL, Skimage, Matplotlib
- **Frameworks & Tools**: Faceapi.js, Google Forms

## Installation and Setup

Clone the repository:
```sh
git clone https://github.com/EmaanBashir/Face-Liveness-Detection.git
```

Create and activate a virtual environment:
```sh
python -m venv venv
```

Navigate to the project directory:
```sh
cd face-liveness-detection
```

Install dependencies:
```sh
pip install -r requirements.txt
```

Create a data directory:
```sh
cd static
mkdir data
cd ..
```

## How to Run
Run the Django server:
```sh
python manage.py runserver
```

Access the application at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Project Demo and Documents
- [Project Proposal](https://drive.google.com/file/d/10XsEFW4dkuzMqnJBK_OzH4cPeDV5LX0X/view?usp=drive_link)
- [Software Requirement Specification](https://drive.google.com/file/d/1tVrbVHQBaxw6Ey1gnsiM6v5T-FkyDGjV/view?usp=drive_link)
- [Software Design Specification](https://drive.google.com/file/d/1m_BqrsAnPXzhYPPIVI0mT33Qp4nN8ti1/view?usp=drive_link)
- [Final Report](https://drive.google.com/file/d/1CDJ6TDsZYAG9MthzJshl4PjAAadm-NAd/view?usp=drive_link)
- [Animated Video](https://drive.google.com/file/d/1jztMXygPjR4qTPDR3tzMTTOUa-YJlNSt/view?usp=drive_link)
- [Final Presentation](https://drive.google.com/file/d/1jIY8cXo8oaZOxds8pcw_OcTCvQ-dJ1Ul/view?usp=drive_link)
- [Project Demo](https://drive.google.com/file/d/1Nb--nPbFbEtHR_-9cuoGmJXtPi090v3d/view?usp=drive_link)
- [Final Poster](https://drive.google.com/file/d/15oILbqB1jOqWerBTctK3aw1q2xuKLBN2/view?usp=drive_link)
- [Standee](https://drive.google.com/file/d/1ymMsnLbR0RE50AmJm-y7iF68gjjbxABu/view?usp=drive_link)
- [Brochures](https://drive.google.com/file/d/1vK4hUfyhGZS9uiR4fjisgnzxm10pUvcC/view?usp=drive_link)
- [Plagiarism Report](https://drive.google.com/file/d/10QbJcK4Lxv9FwvSYDHvkdLoKq4ombmJV/view?usp=drive_link)

## Project Architecture
The system follows a **waterfall model**, ensuring sequential and structured implementation. Below is the high-level architecture:

```
User Interface → Face Detection → Feature Extraction → Liveness Detection → Classification → Result Display
```

## Future Work
- Expand dataset for improved accuracy.
- Implement **3D spoof detection**.
- Incorporate **neural network-based liveness detection**.
- Enhance UI/UX for better user experience.

## User Interface
### Home Page
![image](https://github.com/user-attachments/assets/ebaba352-ebb5-42ee-a6f7-8c2a01e8ae89)

### Demonstration Page
![image](https://github.com/user-attachments/assets/8d8eec8e-10af-4c84-a71a-cb9043698ab1)

### Services Page
![image](https://github.com/user-attachments/assets/e75a68db-4208-48fb-a638-3f0049a5de2a)

### Use Case Page
![image](https://github.com/user-attachments/assets/9fd9e016-06de-4fbe-9778-92dac8e80b1c)

### About Page
![image](https://github.com/user-attachments/assets/8623fabc-5b2e-4b0b-a1ff-68ffd7233452)

### Contact Page
![image](https://github.com/user-attachments/assets/ef5bf1cb-d2ed-4f34-9cbf-e5903ab63a44)

---
> **Note:** This repository is part of my BEng Software Engineering final year project at NUST SEECS.


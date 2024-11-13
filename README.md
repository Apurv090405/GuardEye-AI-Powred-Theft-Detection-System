<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GuardEye: AI-Powered Theft Detection System</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #f4f4f9;
      color: #333;
      line-height: 1.6;
    }
    .container {
      width: 80%;
      margin: auto;
      overflow: hidden;
      padding: 2rem;
    }
    h1, h2, h3 {
      color: #333;
      border-bottom: 2px solid #333;
      padding-bottom: 0.5rem;
    }
    p {
      margin: 1rem 0;
    }
    code, pre {
      background-color: #f4f4f9;
      border-left: 4px solid #333;
      padding: 1rem;
      display: block;
      white-space: pre-wrap;
    }
    ul {
      list-style: none;
      padding: 0;
    }
    ul li {
      margin: 0.5rem 0;
      padding-left: 1rem;
      text-indent: -1rem;
    }
    .code-box {
      background: #272822;
      color: #f8f8f2;
      padding: 1.5rem;
      border-radius: 5px;
      font-family: monospace;
    }
    .highlight {
      color: #66d9ef;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>GuardEye: AI-Powered Theft Detection System</h1>
    <p><strong>GuardEye</strong> is an innovative AI-powered theft detection system aimed at improving security in public spaces. Built using deep learning and real-time video analytics, GuardEye detects suspicious behavior and theft activities with high accuracy, leveraging Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and the NVIDIA Jetson AGX Xavier for real-time processing.</p>
    
    <h2>Table of Contents</h2>
    <ul>
      <li><a href="#project-overview">Project Overview</a></li>
      <li><a href="#features">Features</a></li>
      <li><a href="#architecture">System Architecture</a></li>
      <li><a href="#technologies-used">Technologies Used</a></li>
      <li><a href="#setup-and-installation">Setup and Installation</a></li>
      <li><a href="#usage">Usage</a></li>
      <li><a href="#contributing">Contributing</a></li>
      <li><a href="#license">License</a></li>
    </ul>
    
    <h2 id="project-overview">Project Overview</h2>
    <p>The <strong>GuardEye</strong> system is designed to help prevent theft in crowded or unattended areas by analyzing CCTV footage in real-time. It captures and processes video feeds, detects potential threats, and generates alerts whenever suspicious activities are identified. This system is optimized for low-latency performance on edge devices like the NVIDIA Jetson AGX Xavier, enabling deployment in settings where rapid response is critical.</p>
    
    <h2 id="features">Features</h2>
    <ul>
      <li>Real-time theft and anomaly detection</li>
      <li>Integration with CCTV systems</li>
      <li>Automated alert mechanisms (buzzers, notifications)</li>
      <li>Edge computing optimization on NVIDIA Jetson AGX Xavier</li>
      <li>Robust model trained on the UCF Crime and custom datasets</li>
    </ul>
    
    <h2 id="architecture">System Architecture</h2>
    <p>The system employs a hybrid deep learning model, combining <strong>CNN</strong> for spatial feature extraction and <strong>LSTM</strong> for temporal analysis, enhancing its ability to detect complex activities. GuardEye processes real-time video streams, feeding them into the model hosted on the Jetson Xavier, which then classifies the observed behavior and triggers an alert if suspicious activity is detected.</p>
    
    <h2 id="technologies-used">Technologies Used</h2>
    <ul>
      <li><strong>Python</strong> for core programming</li>
      <li><strong>Keras</strong> and <strong>TensorFlow</strong> for deep learning</li>
      <li><strong>OpenCV</strong> for image processing</li>
      <li><strong>NVIDIA Jetson AGX Xavier</strong> for edge computing</li>
      <li><strong>UCF Crime Dataset</strong> for training and testing</li>
    </ul>
    
    <h2 id="setup-and-installation">Setup and Installation</h2>
    <h3>Prerequisites</h3>
    <ul>
      <li>NVIDIA Jetson AGX Xavier with CUDA and cuDNN installed</li>
      <li>Python 3.8+</li>
      <li>Packages: <code>pip install tensorflow keras opencv-python</code></li>
    </ul>
    
    <h3>Installation Steps</h3>
    <pre class="code-box"><code>git clone https://github.com/yourusername/GuardEye.git
cd GuardEye
pip install -r requirements.txt</code></pre>
    
    <h2 id="usage">Usage</h2>
    <p>To start the GuardEye system, run the following command:</p>
    <pre class="code-box"><code>python main.py</code></pre>
    <p>This will initialize the model, connect to the CCTV feed, and begin real-time monitoring. Alerts will be triggered for any detected suspicious activities, which can be customized based on the deployment requirements.</p>
    
    <h2 id="contributing">Contributing</h2>
    <p>Contributions to GuardEye are welcome! Please fork this repository, make changes, and submit a pull request. Be sure to follow coding standards and document any significant changes.</p>
    
    <h2 id="license">License</h2>
    <p>This project is licensed under the MIT License - see the <code>LICENSE</code> file for details.</p>
    
    <footer>
      <p>&copy; 2024 GuardEye Project Team. All rights reserved.</p>
    </footer>
  </div>
</body>
</html>

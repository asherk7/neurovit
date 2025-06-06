document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    // Predict Tumor
    const predictRes = await fetch('/predict/', {
        method: 'POST',
        body: formData
    });

    const result = await predictRes.json();
    document.getElementById('predictionText').textContent = `Tumor Type: ${result.prediction}`;

    const originalImg = document.getElementById('originalImage');
    const boxImg = document.getElementById('boxImage');
    const imagesContainer = document.querySelector('.images-container');

    originalImg.src = `data:image/png;base64,${result.original_image}`;
    boxImg.src = `data:image/png;base64,${result.box_image}`;

    imagesContainer.style.display = 'flex';

    // Ask AI Doctor
    const chatRes = await fetch('/query/', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: `What does a ${result.prediction} mean?` })
    });

    const chatData = await chatRes.json();
    document.getElementById('chatbotResponse').textContent = chatData.response;
});

document.getElementById('chat-form').addEventListener('submit', async function (e) {
  e.preventDefault();

  const input = document.getElementById('user-question');
  const chatWindow = document.getElementById('chat-window');
  const question = input.value.trim();
  if (!question) return;

  // Add user message to chat
  const userMsg = document.createElement('div');
  userMsg.className = 'message user-message';
  userMsg.textContent = question;
  chatWindow.appendChild(userMsg);

  input.value = '';

  // Scroll to bottom
  chatWindow.scrollTop = chatWindow.scrollHeight;

  // Send request to backend
  const response = await fetch('/chat/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question })
  });

  const data = await response.json();
  const botMsg = document.createElement('div');
  botMsg.className = 'message bot-message';
  botMsg.textContent = data.answer || 'Something went wrong.';
  chatWindow.appendChild(botMsg);

  // Scroll to bottom
  chatWindow.scrollTop = chatWindow.scrollHeight;
});


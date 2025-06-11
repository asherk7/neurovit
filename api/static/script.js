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
    const prediction = result.prediction;
    document.getElementById('predictionText').textContent = `Tumor Type: ${prediction}`;

    // Update Images
    const originalImg = document.getElementById('originalImage');
    const boxImg = document.getElementById('boxImage');
    const imagesContainer = document.querySelector('.images-container');

    originalImg.src = `data:image/png;base64,${result.original_image}`;
    boxImg.src = `data:image/png;base64,${result.box_image}`;
    imagesContainer.style.display = 'flex';

    // First message in chat with explanation
    const chatWindow = document.getElementById('chat-window');

    const userMsg = document.createElement('div');
    userMsg.className = 'message user-message';
    let formattedPrediction = prediction.toLowerCase();
    if (formattedPrediction !== "no tumor") {
      formattedPrediction = "a " + formattedPrediction;
    }
    userMsg.textContent = `According to the MRI brain scan, the image analyzed has ${formattedPrediction}, could you elaborate on what it is?`;
    chatWindow.appendChild(userMsg)

    const chatRes = await fetch('/chat/', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg.textContent })
    });

    const chatData = await chatRes.json();
    const botMsg = document.createElement('div');
    botMsg.className = 'message bot-message';
    botMsg.textContent = chatData.answer || 'Sorry, I couldn’t find information.';

    chatWindow.appendChild(botMsg);

    chatWindow.scrollTop = chatWindow.scrollHeight;
});

document.getElementById('chat-form').addEventListener('submit', async function (e) {
  e.preventDefault();

  const input = document.getElementById('user-question');
  const chatWindow = document.getElementById('chat-window');
  const question = input.value.trim();
  if (!question) return;

  const userMsg = document.createElement('div');
  userMsg.className = 'message user-message';
  userMsg.textContent = question;
  chatWindow.appendChild(userMsg);

  input.value = '';

  chatWindow.scrollTop = chatWindow.scrollHeight;

  const response = await fetch('/chat/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question: question })
  });

  const data = await response.json();
  const botMsg = document.createElement('div');
  botMsg.className = 'message bot-message';
  botMsg.textContent = data.answer || 'Something went wrong.';
  chatWindow.appendChild(botMsg);

  // Scroll to bottom
  chatWindow.scrollTop = chatWindow.scrollHeight;
});

const dropdownBtn = document.querySelector('.dropdown-toggle');
const dropdownMenu = document.querySelector('.dropdown-menu');
const previewBox = document.getElementById('previewBox');
const previewImage = document.getElementById('previewImage');

dropdownBtn.addEventListener('click', (e) => {
  e.preventDefault();
  dropdownMenu.style.display = dropdownMenu.style.display === 'block' ? 'none' : 'block';
});

dropdownMenu.querySelectorAll('li').forEach(item => {
  item.addEventListener('mouseover', () => {
    previewImage.src = item.getAttribute('data-img');
    previewBox.style.display = 'block';
  });

  item.addEventListener('mouseout', () => {
    previewBox.style.display = 'none';
  });

  item.addEventListener('click', async () => {
    const imgPath = item.getAttribute('data-img');

    // Fetch image and convert to File object
    const response = await fetch(imgPath);
    const blob = await response.blob();
    const file = new File([blob], imgPath.split('/').pop(), { type: blob.type });

    // Set to file input
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    document.getElementById('fileInput').files = dataTransfer.files;

    dropdownMenu.style.display = 'none';
    previewBox.style.display = 'none';
  });
});

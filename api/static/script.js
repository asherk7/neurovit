// Handle form submission for tumor prediction
document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    // Send image to backend /predict endpoint
    const predictRes = await fetch('/predict/', {
        method: 'POST',
        body: formData
    });

    const result = await predictRes.json();
    const prediction = result.prediction;

    // Show prediction text on page
    document.getElementById('predictionText').textContent = `Tumor Type: ${prediction}`;

    // Update displayed images (original + heatmap)
    const originalImg = document.getElementById('originalImage');
    const boxImg = document.getElementById('boxImage');
    const imagesContainer = document.querySelector('.images-container');

    originalImg.src = `data:image/png;base64,${result.original_image}`;
    boxImg.src = `data:image/png;base64,${result.box_image}`;
    imagesContainer.style.display = 'flex';

    // Add first user message to chat window based on prediction
    const chatWindow = document.getElementById('chat-window');

    const userMsg = document.createElement('div');
    userMsg.className = 'message user-message';
    let formattedPrediction = prediction.toLowerCase();
    if (formattedPrediction !== "no tumor") {
      formattedPrediction = "a " + formattedPrediction;
    }
    userMsg.textContent = `According to the MRI brain scan, the image analyzed has ${formattedPrediction}, could you elaborate on what it is?`;
    chatWindow.appendChild(userMsg)

    // Send message to backend /chat endpoint to get AI response
    const chatRes = await fetch('/chat/', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg.textContent })
    });

    const chatData = await chatRes.json();

    // Add AI response to chat window
    const botMsg = document.createElement('div');
    botMsg.className = 'message bot-message';
    botMsg.textContent = chatData.answer || 'Sorry, I couldn’t find information.';

    chatWindow.appendChild(botMsg);

    // Scroll chat to bottom to show new messages
    chatWindow.scrollTop = chatWindow.scrollHeight;
});

// Handle manual chat question submission
document.getElementById('chat-form').addEventListener('submit', async function (e) {
  e.preventDefault();

  const input = document.getElementById('user-question');
  const chatWindow = document.getElementById('chat-window');
  const question = input.value.trim();
  if (!question) return;

  // Add user message to chat window
  const userMsg = document.createElement('div');
  userMsg.className = 'message user-message';
  userMsg.textContent = question;
  chatWindow.appendChild(userMsg);

  input.value = '';

  // Scroll chat to bottom
  chatWindow.scrollTop = chatWindow.scrollHeight;

  // Send question to backend /chat endpoint
  const response = await fetch('/chat/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question: question })
  });

  const data = await response.json();

  // Add AI response to chat window
  const botMsg = document.createElement('div');
  botMsg.className = 'message bot-message';
  botMsg.textContent = data.answer || 'Something went wrong.';
  chatWindow.appendChild(botMsg);

  // Scroll chat to bottom
  chatWindow.scrollTop = chatWindow.scrollHeight;
});

// Dropdown toggle button behavior
const dropdownBtn = document.querySelector('.dropdown-toggle');
const dropdownMenu = document.querySelector('.dropdown-menu');
const previewBox = document.getElementById('previewBox');
const previewImage = document.getElementById('previewImage');

dropdownBtn.addEventListener('click', (e) => {
  e.preventDefault();
  // Toggle dropdown menu visibility
  dropdownMenu.style.display = dropdownMenu.style.display === 'block' ? 'none' : 'block';
});

// Handle hover and click on dropdown items
dropdownMenu.querySelectorAll('li').forEach(item => {
  // Show preview image on hover
  item.addEventListener('mouseover', () => {
    previewImage.src = item.getAttribute('data-img');
    previewBox.style.display = 'block';
  });

  // Hide preview on mouse out
  item.addEventListener('mouseout', () => {
    previewBox.style.display = 'none';
  });

  // On item click, fetch image and set it in file input
  item.addEventListener('click', async () => {
    const imgPath = item.getAttribute('data-img');

    // Fetch image as blob
    const response = await fetch(imgPath);
    const blob = await response.blob();

    // Create a File object from blob
    const file = new File([blob], imgPath.split('/').pop(), { type: blob.type });

    // Create a DataTransfer to set the file input programmatically
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    document.getElementById('fileInput').files = dataTransfer.files;

    // Hide dropdown and preview after selection
    dropdownMenu.style.display = 'none';
    previewBox.style.display = 'none';
  });
});

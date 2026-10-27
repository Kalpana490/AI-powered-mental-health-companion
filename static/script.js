document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("user-input");
  const button = document.getElementById("send-btn");

  // When the button is clicked
  button.addEventListener("click", sendMessage);

  // When Enter key is pressed
  input.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  });
});

function appendMessage(sender, message) {
  const chat = document.getElementById("chat-messages");
  const msg = document.createElement("p");
  msg.className = sender === "bot" ? "bot-message" : "user-message";
  msg.textContent = message;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  appendMessage("user", message);
  input.value = "";
  appendMessage("bot", "...");

  fetch("/chat_api", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  })
  .then(res => res.json())
  .then(data => {
    const lastBotMsg = document.querySelector(".bot-message:last-child");
    lastBotMsg.textContent = data.response;
  })
  .catch(err => {
    const lastBotMsg = document.querySelector(".bot-message:last-child");
    lastBotMsg.textContent = "❌ Failed to reach server.";
    console.error(err);
  });
}

// frontend/app.js
const base = "http://localhost:8000";

async function upload(){
  const f = document.getElementById("pdffile").files[0];
  if(!f) return alert("Choose a PDF");
  const fd = new FormData();
  fd.append("file", f);
  const res = await fetch(base + "/upload_pdf/", {method:"POST", body: fd});
  const j = await res.json();
  alert("Processed: " + JSON.stringify(j));
}

async function getSummary(){
  const res = await fetch(base + "/summary/");
  const j = await res.json();
  document.getElementById("summary").textContent = j.summary;
}

async function ask(){
  const q = document.getElementById("q").value;
  if(!q) return;
  const res = await fetch(base + "/query/", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({query: q})
  });
  const j = await res.json();
  const box = document.getElementById("chatbox");
  box.innerHTML += `<div><b>You:</b> ${q}</div>`;
  box.innerHTML += `<div><b>Bot:</b> ${j.answer}</div><hr/>`;
  box.scrollTop = box.scrollHeight;
}

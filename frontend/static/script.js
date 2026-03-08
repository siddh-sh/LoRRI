const API = "https://lorri.onrender.com";

async function analyzeShipment(data) {

  const response = await fetch(`${API_BASE}/api/shipment/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  console.log(result);
}
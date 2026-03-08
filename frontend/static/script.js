const API_BASE = "https://lorri.onrender.com";

async function analyzeShipment() {

  const laneId = "L001";      // example
  const weight = 1000;        // example

  try {
    const response = await fetch(`${API_BASE}/api/shipment/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        lane_id: laneId,
        weight_kg: weight
      })
    });

    const data = await response.json();
    console.log(data);

  } catch (error) {
    console.error("Analysis failed", error);
  }
}
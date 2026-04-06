const handleAnalyze = async () => {
  if (!uploadedFile) return;

  setLoading(true);
  setError(null);

  try {
    const formData = new FormData();
    formData.append('file', uploadedFile);

    // 🔥 Wake up Render (VERY IMPORTANT)
    await fetch("https://ecg-detector-1.onrender.com/");

    const response = await axios.post(
      "https://ecg-detector-1.onrender.com/api/analyze",
      formData,
      {
        timeout: 120000 // increase timeout for ML model
      }
    );

    setResult(response.data);

  } catch (err) {
    console.error("FULL ERROR:", err);

    const msg =
      err.response?.data?.detail ||
      err.message ||
      "Analysis failed. Please try again.";

    setError(msg);
  } finally {
    setLoading(false);
  }
};
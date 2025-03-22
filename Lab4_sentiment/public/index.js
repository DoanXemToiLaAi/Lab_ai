document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');

    analyzeBtn.addEventListener('click', async () => {
        const inputText = document.getElementById('inputText').value;
        console.log("Gửi yêu cầu với text:", inputText);

        try {
            const response = await fetch('/sentiment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: inputText })
            });
            const data = await response.json();
            console.log("Dữ liệu nhận được:", data);

            let result = data.result;

            // Nếu nhận được mảng kết quả, lấy kết quả có độ tin cậy cao nhất
            if (Array.isArray(result) && result.length > 1) {
                result.sort((a, b) => b.score - a.score);
                result = result[0];
            } else if (Array.isArray(result)) {
                result = result[0];
            }

            // Chọn icon hiển thị dựa trên nhãn sentiment
            let icon = "";
            if (result.label === "POSITIVE") {
                icon = "👍";
            } else if (result.label === "NEGATIVE") {
                icon = "👎";
            } else {
                icon = "😐";
            }

            // Hiển thị kết quả (label, score và icon)
            document.getElementById('result').innerHTML = `
                <div style="font-size: 1.5em; display: flex; align-items: center;">
                    <span style="margin-right: 10px;">${icon}</span>
                    <span>${result.label} (${(result.score * 100).toFixed(2)}%)</span>
                </div>
            `;
        } catch (error) {
            console.error("Lỗi khi gọi API:", error);
            document.getElementById('result').innerText = 'Có lỗi xảy ra: ' + error.message;
        }
    });
});

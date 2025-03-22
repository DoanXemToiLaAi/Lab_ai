function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    const file = fileInput.files[0];

    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            console.log("Kết quả OCR:", data.prediction);

            // Hiển thị kết quả nhận diện chữ
            document.getElementById('result').textContent = data.prediction[0]?.generated_text || "Không có văn bản";

            // Hiển thị ảnh đã tải lên
            const imgPreview = document.getElementById('preview');
            imgPreview.src = URL.createObjectURL(file);
            imgPreview.style.display = 'block';
        })
        .catch(error => {
            console.error('Lỗi:', error);
        });
}

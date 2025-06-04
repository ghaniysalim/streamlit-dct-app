import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# --- Fungsi DCT dan IDCT (menggunakan OpenCV) ---
def apply_dct(block):
    """Menghitung DCT 2D pada blok."""
    return cv2.dct(block.astype(np.float32))

def apply_idct(dct_block):
    """Menghitung IDCT 2D pada blok DCT."""
    return cv2.idct(dct_block)

# --- Matriks Kuantisasi Default (JPEG Luminance untuk 8x8) ---
DEFAULT_QUANT_MATRIX_8X8 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def get_scaled_quantization_matrix(base_matrix, target_size, quality_factor=None):
    """
    Menskala matriks kuantisasi ke ukuran target atau berdasarkan faktor kualitas.
    Jika quality_factor diberikan, matriks default 8x8 diskalakan.
    Jika tidak, matriks dasar diskalakan ke ukuran target.
    """
    if quality_factor is not None:
        if quality_factor < 50:
            scale_factor = 5000 / quality_factor
        else:
            scale_factor = 200 - 2 * quality_factor
        scaled_matrix = np.floor((base_matrix * scale_factor + 50) / 100)
        scaled_matrix = np.clip(scaled_matrix, 1, 255)
        return scaled_matrix.astype(np.float32)

    if base_matrix.shape[0] == target_size:
        return base_matrix.astype(np.float32)
    
    scaled_matrix = cv2.resize(base_matrix, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scaled_matrix = np.clip(np.round(scaled_matrix), 1, 255)
    return scaled_matrix.astype(np.float32)

# --- Fungsi-fungsi terpisah untuk setiap proses pada blok ---
def _apply_dct_to_block(block_data):
    """Melakukan level shifting (-128) dan DCT 2D pada blok."""
    block_shifted = block_data.astype(np.float32) - 128
    dct_coeffs = apply_dct(block_shifted)
    return dct_coeffs

def _quantize_block(dct_coeffs, quantization_matrix):
    """Melakukan kuantisasi pada koefisien DCT."""
    quantized_coeffs = np.round(dct_coeffs / quantization_matrix)
    return quantized_coeffs

def _truncate_coefficients(quantized_coeffs, block_size, num_coeffs_to_keep):
    """
    Mempertahankan hanya koefisien DCT teratas (berdasarkan urutan zigzag)
    dan mengatur sisanya menjadi nol.
    """
    truncated_coeffs = np.copy(quantized_coeffs)
    
    # Generate zigzag order
    zigzag_order = []
    row, col = 0, 0
    direction = 1
    for _ in range(block_size * block_size):
        zigzag_order.append((row, col))
        if direction == 1:
            if col == block_size - 1: row += 1; direction = -1
            elif row == 0: col += 1; direction = -1
            else: row -= 1; col += 1
        else:
            if row == block_size - 1: col += 1; direction = 1
            elif col == 0: row += 1; direction = 1
            else: row += 1; col -= 1
    
    for k in range(num_coeffs_to_keep, block_size * block_size):
        r, c = zigzag_order[k]
        truncated_coeffs[r, c] = 0
            
    return truncated_coeffs

def _dequantize_block(quantized_coeffs, quantization_matrix):
    """Melakukan dekuantisasi pada koefisien yang dikuantisasi."""
    dequantized_coeffs = quantized_coeffs * quantization_matrix
    return dequantized_coeffs

def _apply_idct_to_block(dequantized_coeffs):
    """Melakukan IDCT 2D dan level shifting kembali (+128) pada blok."""
    reconstructed_block = apply_idct(dequantized_coeffs) + 128
    reconstructed_block = np.clip(reconstructed_block, 0, 255)
    return reconstructed_block.astype(np.uint8)

# --- Fungsi Zigzag Scan dan Inverse Zigzag Scan (untuk flattening/restoring) ---
def _zigzag_scan(block, N):
    """
    Mengubah blok 2D NxN menjadi array 1D menggunakan pola zigzag.
    """
    result = []
    row, col = 0, 0
    direction = 1

    for _ in range(N * N):
        result.append(block[row][col])

        if direction == 1:
            if col == N - 1:
                row += 1
                direction = -1
            elif row == 0:
                col += 1
                direction = -1
            else:
                row -= 1
                col += 1
        else:
            if row == N - 1:
                col += 1
                direction = 1
            elif col == 0:
                row += 1
                direction = 1
            else:
                row += 1
                col -= 1
    return result

def _inverse_zigzag_scan(array, N):
    """
    Mengubah array 1D kembali menjadi blok 2D NxN menggunakan pola inverse zigzag.
    """
    block = np.zeros((N, N), dtype=np.float32)
    row, col = 0, 0
    direction = 1

    for i in range(N * N):
        block[row][col] = array[i]

        if direction == 1:
            if col == N - 1:
                row += 1
                direction = -1
            elif row == 0:
                col += 1
                direction = -1
            else:
                row -= 1
                col += 1
        else:
            if row == N - 1:
                col += 1
                direction = 1
            elif col == 0:
                row += 1
                direction = 1
            else:
                row += 1
                col -= 1
    return block

# --- Fungsi Run-Length Encoding (RLE) dan Decoding ---
def _run_length_encode(data):
    """
    Melakukan Run-Length Encoding (RLE) sederhana pada array 1D.
    """
    encoded = []
    if not data:
        return encoded

    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            if count > 1:
                encoded.append((data[i - 1], count))
            else:
                encoded.append(data[i - 1])
            count = 1
    # Tambahkan elemen/run terakhir
    if count > 1:
        encoded.append((data[len(data) - 1], count))
    else:
        encoded.append(data[len(data) - 1])
    return encoded

def _run_length_decode(encoded_data):
    """
    Melakukan Run-Length Decoding (RLE) pada data yang dienkode.
    """
    decoded = []
    for item in encoded_data:
        if isinstance(item, tuple) and len(item) == 2:
            value, count = item
            decoded.extend([value] * count)
        else:
            decoded.append(item)
    return decoded

def process_image_channel(channel_data, block_size, quantization_matrix, num_coeffs_to_keep):
    """
    Memproses satu saluran gambar (Y, Cb, atau Cr) melalui pipeline DCT.
    Mengembalikan saluran yang direkonstruksi dan koefisien kuantisasi dari blok pertama.
    """
    height, width = channel_data.shape
    reconstructed_channel = np.zeros_like(channel_data, dtype=np.uint8)
    first_block_quant_coeffs = None

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = channel_data[i:i+block_size, j:j+block_size]

            dct_coeffs = _apply_dct_to_block(block)
            truncated_dct_coeffs = _truncate_coefficients(dct_coeffs, block_size, num_coeffs_to_keep)
            quantized_coeffs = _quantize_block(truncated_dct_coeffs, quantization_matrix)
            
            if i == 0 and j == 0:
                first_block_quant_coeffs = quantized_coeffs

            zigzagged_data = _zigzag_scan(quantized_coeffs, block_size)
            encoded_data = _run_length_encode(zigzagged_data)
            decoded_data = _run_length_decode(encoded_data)
            
            if len(decoded_data) != block_size * block_size:
                decoded_data.extend([0] * (block_size * block_size - len(decoded_data)))

            dezigzagged_coeffs = _inverse_zigzag_scan(decoded_data, block_size)
            dequantized_coeffs = _dequantize_block(dezigzagged_coeffs, quantization_matrix)
            reconstructed_block = _apply_idct_to_block(dequantized_coeffs)
            
            reconstructed_channel[i:i+block_size, j:j+block_size] = reconstructed_block
    
    return reconstructed_channel, first_block_quant_coeffs

# --- Aplikasi Streamlit ---
st.set_page_config(layout="wide", page_title="Kompresi Gambar DCT Interaktif")

st.title("Kompresi Gambar DCT Interaktif")
st.write("Unggah gambar dan sesuaikan parameter kompresi DCT.")

# --- Sidebar untuk Kontrol ---
st.sidebar.header("Kontrol Kompresi")

uploaded_file = st.sidebar.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# Karena hanya 8x8 yang berfungsi, kita batasi opsinya
block_size = st.sidebar.selectbox("Ukuran Blok", options=[8], index=0)

quant_choice = st.sidebar.selectbox(
    "Matriks Kuantisasi",
    options=['Default JPEG Luminance', 'Faktor Kualitas', 'Matriks Kustom'],
    index=0
)

quality_factor = st.sidebar.slider(
    "Kualitas (1-100)",
    min_value=1,
    max_value=100,
    value=50,
    disabled=(quant_choice != 'Faktor Kualitas')
)

custom_quant_matrix_str = st.sidebar.text_area(
    "Matriks Kustom (pisahkan spasi/koma)",
    value="",
    placeholder=f'Masukkan {block_size*block_size} angka dipisahkan spasi/koma (untuk {block_size}x{block_size} blok)',
    disabled=(quant_choice != 'Matriks Kustom')
)

num_coeffs_to_keep = st.sidebar.slider(
    "Jumlah Koefisien (untuk rekonstruksi)",
    min_value=1,
    max_value=block_size * block_size, # Max 64 for 8x8
    value=block_size * block_size,
)

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Gagal memuat gambar. Pastikan file gambar valid.")
    else:
        st.info("Memproses gambar...")

        # Pastikan dimensi gambar adalah kelipatan dari block_size
        h, w, _ = img.shape
        if h % block_size != 0 or w % block_size != 0:
            st.warning(f"Mengubah ukuran gambar dari {w}x{h} agar kelipatan {block_size}x{block_size}...")
            new_h = (h // block_size) * block_size
            new_w = (w // block_size) * block_size
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape
            st.info(f"Ukuran gambar baru: {w}x{h}")

        # Dapatkan matriks kuantisasi berdasarkan pilihan pengguna
        quantization_matrix = None
        if quant_choice == 'Default JPEG Luminance':
            quantization_matrix = get_scaled_quantization_matrix(DEFAULT_QUANT_MATRIX_8X8, block_size)
        elif quant_choice == 'Faktor Kualitas':
            quantization_matrix = get_scaled_quantization_matrix(DEFAULT_QUANT_MATRIX_8X8, block_size, quality_factor=quality_factor)
        elif quant_choice == 'Matriks Kustom':
            try:
                values = np.array(list(map(int, custom_quant_matrix_str.replace(',', ' ').split()))).reshape(block_size, block_size)
                if np.any(values <= 0):
                    st.error("Nilai matriks kustom harus positif.")
                    return
                quantization_matrix = values.astype(np.float32)
            except ValueError:
                st.error(f"Input matriks kustom tidak valid. Pastikan Anda memasukkan {block_size*block_size} angka.")
                return
            except Exception as e:
                st.error(f"Error saat memproses matriks kustom: {e}")
                return
        
        if quantization_matrix is None:
            st.error("Matriks kuantisasi tidak dapat ditentukan.")
            st.stop()

        # Konversi BGR ke YCrCb (OpenCV menggunakan BGR secara default)
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 

        Y_channel = img_ycbcr[:,:,0]
        Cr_channel = img_ycbcr[:,:,1]
        Cb_channel = img_ycbcr[:,:,2]

        # --- Hitung nilai rata-rata RGB dan YCbCr untuk gambar asli ---
        original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        avg_original_r, avg_original_g, avg_original_b = np.mean(original_rgb[:,:,0]), np.mean(original_rgb[:,:,1]), np.mean(original_rgb[:,:,2])
        avg_original_y, avg_original_cr, avg_original_cb = np.mean(Y_channel), np.mean(Cr_channel), np.mean(Cb_channel)

        st.subheader("Informasi Gambar")
        st.write(f"**--- Nilai Rata-rata Gambar Asli ---**")
        st.write(f"RGB: R={avg_original_r:.2f}, G={avg_original_g:.2f}, B={avg_original_b:.2f}")
        st.write(f"YCrCb: Y={avg_original_y:.2f}, Cr={avg_original_cr:.2f}, Cb={avg_original_cb:.2f}")


        # Proses setiap saluran dan dapatkan koefisien kuantisasi dari blok pertama Y
        reconstructed_Y, first_y_quant_coeffs_block = process_image_channel(Y_channel, block_size, quantization_matrix, num_coeffs_to_keep)
        reconstructed_Cr, _ = process_image_channel(Cr_channel, block_size, quantization_matrix, num_coeffs_to_keep)
        reconstructed_Cb, _ = process_image_channel(Cb_channel, block_size, quantization_matrix, num_coeffs_to_keep)

        # Gabungkan saluran yang direkonstruksi
        reconstructed_ycbcr = cv2.merge([reconstructed_Y, reconstructed_Cr, reconstructed_Cb])
        reconstructed_img = cv2.cvtColor(reconstructed_ycbcr, cv2.COLOR_YCrCb2BGR)

        # --- Hitung nilai rata-rata RGB dan YCbCr untuk gambar rekonstruksi ---
        reconstructed_rgb = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB)
        avg_reconstructed_r, avg_reconstructed_g, avg_reconstructed_b = np.mean(reconstructed_rgb[:,:,0]), np.mean(reconstructed_rgb[:,:,1]), np.mean(reconstructed_rgb[:,:,2])
        avg_reconstructed_y, avg_reconstructed_cr, avg_reconstructed_cb = np.mean(reconstructed_Y), np.mean(reconstructed_Cr), np.mean(reconstructed_Cb)

        st.write(f"**--- Nilai Rata-rata Gambar Rekonstruksi ---**")
        st.write(f"RGB: R={avg_reconstructed_r:.2f}, G={avg_reconstructed_g:.2f}, B={avg_reconstructed_b:.2f}")
        st.write(f"YCrCb: Y={avg_reconstructed_y:.2f}, Cr={avg_reconstructed_cr:.2f}, Cb={avg_reconstructed_cb:.2f}")


        # Hitung PSNR (untuk saluran Y saja)
        mse_y = np.mean((Y_channel - reconstructed_Y)**2)
        max_pixel_value = 255
        psnr_y = 10 * np.log10((max_pixel_value**2) / mse_y) if mse_y != 0 else float('inf')
        
        st.write(f"**PSNR (Y-channel):** {psnr_y:.2f} dB")

        # --- Hitung Rasio Kompresi (berdasarkan koefisien non-nol) ---
        total_coeffs_per_block = block_size * block_size
        non_zero_coeffs_ratio = num_coeffs_to_keep / total_coeffs_per_block
        st.write(f"**Rasio Koefisien Non-Nol yang Dipertahankan:** {non_zero_coeffs_ratio:.2f} ({num_coeffs_to_keep} dari {total_coeffs_per_block})")


        # --- Visualisasi Hasil (Layout 2x2) ---
        st.subheader("Visualisasi Hasil")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Gambar Asli
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Gambar Asli')
        axes[0, 0].axis('off')

        # Matriks Kuantisasi yang Digunakan
        axes[0, 1].imshow(quantization_matrix, cmap='viridis')
        axes[0, 1].set_title('Matriks Kuantisasi Digunakan')
        axes[0, 1].axis('off')
        for (j, i), label in np.ndenumerate(quantization_matrix):
            axes[0, 1].text(i, j, int(label), ha='center', va='center', color='white' if label < np.mean(quantization_matrix) else 'black', fontsize=8)


        # Koefisien DCT Pertama (Y-Channel, setelah kuantisasi & pemotongan)
        axes[1, 0].imshow(np.log(1 + np.abs(first_y_quant_coeffs_block)), cmap='gray')
        axes[1, 0].set_title('Koefisien DCT Blok Pertama (Y-Channel, Setelah Kuantisasi & Pemotongan)')
        axes[1, 0].axis('off')

        # Gambar Rekonstruksi Akhir
        axes[1, 1].imshow(cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Gambar Rekonstruksi (Blok {block_size}x{block_size}, Koef. {num_coeffs_to_keep}, PSNR: {psnr_y:.2f} dB)')
        axes[1, 1].axis('off')

        st.pyplot(fig) # Menampilkan plot Matplotlib di Streamlit
        plt.close(fig) # Penting untuk menutup figure agar tidak memakan memori
else:
    st.info("Silakan unggah gambar di sidebar untuk memulai.")


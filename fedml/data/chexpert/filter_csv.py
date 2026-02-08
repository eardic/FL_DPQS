import csv
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Etiket başlıkları
label_header = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

if __name__ == '__main__':

    label_path = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\chexpert\train.csv"  # Orijinal CSV dosyanızın yolu
    train_output_path = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\chexpert\train_pneumonia.csv"  # Yeni CSV dosyasının yolu
    test_output_path = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\chexpert\test_pneumonia.csv"  # Yeni CSV dosyasının yolu

    images = []
    labels = []

    # Orijinal CSV'den verileri yükleme
    with open(label_path, "r") as f:
        reader = csv.reader(f)
        # Başlık satırını atla
        _ = next(reader)
        for row in reader:
            _img = row[0]
            _label = row[5:]  # Etiket sütunlarını al
            for i in range(len(_label)):
                if _label[i] == "":  # Boş etiketleri negatif olarak kabul et
                    _label[i] = 0
                else:
                    _label[i] = int(float(_label[i]))  # Etiketleri integer olarak dönüştür
            images.append(_img)
            labels.append(_label)

    # Verileri DataFrame olarak yükleme
    df = pd.DataFrame(labels, columns=label_header)
    df["Image"] = images  # Görüntü adlarını ekle

    # Pneumonia için pozitif, negatif ve belirsiz örnekleri ayır
    df_pneumonia = df[["Image", "Pneumonia"]]

    # Belirsizleri 2 olarak etiketle
    df_pneumonia["Pneumonia"] = df_pneumonia["Pneumonia"].apply(lambda x: 2 if x == -1 else x)

    # Belirsiz ve pozitif örnekler
    positive_samples = df_pneumonia[df_pneumonia["Pneumonia"] == 1]
    uncertain_samples = df_pneumonia[df_pneumonia["Pneumonia"] == 2]

    # Negatif örnekler
    negative_samples = df_pneumonia[df_pneumonia["Pneumonia"] == 0]

    # Pneumonia dışındaki her bir etiketten belirli sayıda negatif örnek seçme
    negative_balanced = []
    samples_per_label = len(uncertain_samples) // (len(label_header) - 1)  # Her etiketten alınacak örnek sayısı

    for label in label_header:
        if label != "Pneumonia":
            neg_samples_for_label = df[(df[label] == 1) & (df["Pneumonia"] == 0)]
            sample_size = min(samples_per_label, len(neg_samples_for_label))
            if sample_size > 0:
                neg_samples_for_label = neg_samples_for_label.sample(n=sample_size, random_state=42)
                negative_balanced.append(neg_samples_for_label)

    # Seçilen negatifleri birleştir
    negative_balanced_df = pd.concat(negative_balanced).drop_duplicates()

    # Negatif sayısını pozitif sayısına eşitlemek için rastgele alt örnekleme
    if len(negative_balanced_df) >= len(uncertain_samples):
        negative_balanced_df = negative_balanced_df.sample(n=len(positive_samples), random_state=42)
    else:
        print(
            f"Uyarı: Negatif örnek sayısı ({len(negative_balanced_df)}) pozitif örneklerden ({len(uncertain_samples)}) daha az. Mevcut negatif örneklerin tamamı kullanılacak.")

    # Tüm sınıfları birleştir
    balanced_df = pd.concat([positive_samples, negative_balanced_df[["Image", "Pneumonia"]], uncertain_samples])
    balanced_df = shuffle(balanced_df).reset_index(drop=True)

    train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df["Pneumonia"])

    # Eğitim ve test sınıf istatistiklerini yazdır
    print("Eğitim Sınıf İstatistikleri:")
    print(train_df["Pneumonia"].value_counts())

    print("\nTest Sınıf İstatistikleri:")
    print(test_df["Pneumonia"].value_counts())

    # Yeni CSV dosyalarını kaydetme
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"CSV dosyaları kaydedildi.")

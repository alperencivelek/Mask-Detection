from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# ilk öğrenme oranını, eğitilecek dönem sayısını
# ve parti boyutunu baslatalim
baslangic_ogrenmesi = 1e-4
ogrenme_devri = 20
parti_boyutu = 16

dizin = os.getcwd()+"\\Dataset\\Train"
kategoriler = ["Mask","Non Mask"]


#veri seti dizinimizdeki görüntülerin listesini alın, ardından veri listesini (yani, görüntüler) ve sınıf görüntülerini başlatın
print("Resimler yukleniyor")

veri = []
etiketler = []

for kategori in kategoriler:
    dosya_yolu = os.path.join(dizin, kategori)
    for resim in os.listdir(dosya_yolu):
        resim_dosya_yolu = os.path.join(dosya_yolu, resim)
        resim = load_img(resim_dosya_yolu, target_size=(224, 224))
        resim = img_to_array(resim)
        resim = preprocess_input(resim)

        veri.append(resim)
        etiketler.append(kategori)

# one-hot encoding yapalim
lb = LabelBinarizer()
etiketler = lb.fit_transform(etiketler)
etiketler = to_categorical(etiketler)

veri = np.array(veri, dtype="float32")
etiketler = np.array(etiketler)

(trainX, testX, trainY, testY) = train_test_split(veri, etiketler,
    test_size=0.20, stratify=etiketler, random_state=42)

# veri büyütme için eğitim görüntüsü oluşturucuyu oluşturun
ImageAugmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Baş FC katman kümelerinin kapalı kalmasını sağlayarak MobileNetV2 ağını yükleyin
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# temel modelin üstüne yerleştirilecek modelin tepesini oluşturun
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# tepe FC modelini temel modelin üstüne yerleştirin (bu, eğiteceğimiz asıl model olacak)
model = Model(inputs=baseModel.input, outputs=headModel)

# temel modeldeki tüm katmanlar üzerinde döngü yapın ve onları dondurun, böylece ilk eğitim sürecinde *güncellenmezler*
for layer in baseModel.layers:
    layer.trainable = False

# modelimizi derle
print("Model Derleniyor")
opt = Adam(learning_rate=baslangic_ogrenmesi, decay=baslangic_ogrenmesi / ogrenme_devri)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# tepe network egitimi
print("Egitim")
H = model.fit(
    ImageAugmentation.flow(trainX, trainY, batch_size=parti_boyutu),
    steps_per_epoch=len(trainX) // parti_boyutu,
    validation_data=(testX, testY),
    validation_steps=len(testX) // parti_boyutu,
    epochs=ogrenme_devri)

# test setinde tahminlerde bulunun
print("Network Degerlendiriliyor")
predIdxs = model.predict(testX, batch_size=parti_boyutu)

# test setindeki her görüntü için dizinin dizinini bulmamız gerekiyor.karşılık gelen en büyük tahmin olasılığına sahip etiket
predIdxs = np.argmax(predIdxs, axis=1)

# güzel biçimlendirilmiş bir sınıflandırma raporu göster
print(classification_report(testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_))

# modeli diske yaz
print("MODEL KAYDEDILIYOR")
model.summary()
print(len(model.layers))
model.save("maske_tespiti.model", save_format="h5")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, ogrenme_devri), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, ogrenme_devri), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, ogrenme_devri), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, ogrenme_devri), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")




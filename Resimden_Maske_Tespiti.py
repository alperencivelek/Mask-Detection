from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os


def mask_image():
    # serileştirilmiş yüz dedektör modelimizi diskten yükleyin
    txtPath = r"face_detector\deploy.txt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    yuzAgi = cv2.dnn.readNet(txtPath, weightsPath)

    # yüz maskesi dedektör modelini diskten yükleyin
    model = load_model("maske_tespiti.model")

    # giriş görüntüsünü diskten yükleyin, klonlayın ve görüntünün uzamsal boyutlarını alın
    resim = cv2.imread("maskedman.jpg")
    org_resim = resim.copy()
    (a, b) = resim.shape[:2]

    # görüntüden bir blob oluştur
    blob = cv2.dnn.blobFromImage(resim, 1.0, (224, 224),(104.0, 177.0, 123.0))

    # blobu ağ üzerinden geçirin ve yüz algılamalarını alın
    yuzAgi.setInput(blob)
    tespitler = yuzAgi.forward()

    # algılamalar üzerinde döngü
    for i in range(0, tespitler.shape[2]):
        # algılamayla ilişkili güveni (yani olasılığı) çıkarın
        guvenilirlik = tespitler[0, 0, i, 2]

        # Güvenin minimum güvenden daha büyük olmasını sağlayarak zayıf algılamaları filtreleyin
        if guvenilirlik > 0.5:
            # nesne için sınırlayıcı kutunun (x, y)-koordinatlarını hesaplayın
            kutu = tespitler[0, 0, i, 3:7] * np.array([b, a, b, a])
            (baslangic1, baslangic2, bitis1, bitis2) = kutu.astype("int")

            # sınırlayıcı kutuların çerçevenin boyutları dahilinde olduğundan emin olun
            (baslangic1, baslangic2) = (max(0, baslangic1), max(0, baslangic2))
            (bitis1, bitis2) = (min(b - 1, bitis1), min(a - 1, bitis2))

            # yüz ROI'sini çıkarın, BGR'den RGB kanalına dönüştürün, 224x224'e yeniden boyutlandırın ve önişleyin
            yuz = resim[baslangic2:bitis2, baslangic1:bitis1]
            yuz = cv2.cvtColor(yuz, cv2.COLOR_BGR2RGB)
            yuz = cv2.resize(yuz, (224, 224))
            yuz = img_to_array(yuz)
            yuz = preprocess_input(yuz)
            yuz = np.expand_dims(yuz, axis=0)

            # yüzün maskeli olup olmadığını belirlemek için yüzü modelden geçirin
            (mask, withoutMask) = model.predict(yuz)[0]

            # sınırlayıcı kutuyu ve metni çizmek için kullanacağımız sınıf etiketini ve rengini belirleyin
            if mask>withoutMask:
                etiket="Maskeli"
            else:
                etiket="Maskesiz"

            if etiket=="Maskeli":
                renk=(0,255,0)
            else:
                renk=(0,0,255)

            # etikete olasılığı dahil et
            etiket = "{}: {:.2f}%".format(etiket, max(mask, withoutMask) * 100)

            # etiket ve sınırlayıcı kutu dikdörtgenini çıktı çerçevesinde görüntüle
            cv2.putText(resim, etiket, (baslangic1, baslangic2 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, renk, 2)
            cv2.rectangle(resim, (baslangic1, baslangic2), (bitis1, bitis2), renk, 2)

    # çıktı görüntüsünü göster
    cv2.imshow("Cıktı", resim)
    cv2.waitKey(0)

mask_image()
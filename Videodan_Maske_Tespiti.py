from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

def maske_tespiti_ve_tahmini(video_kare, yuzAgi, maskeAgi):
	# Video karesinin boyutlari alinarak blob inşa edilir
	(a, b) = video_kare.shape[:2]
	blob = cv2.dnn.blobFromImage(video_kare, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# blobu ağ üzerinden geçirin ve yüz algılamalarını alın
	yuzAgi.setInput(blob)
	tespitler = yuzAgi.forward()
	print(tespitler.shape)

	# yüzler listemizi, bunlara karşılık gelen konumları ve yüz maskesi ağımızdan tahminlerin listesini başlat
	yuzler = []
	yerler = []
	tahminler = []

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

			# yüz ROI'sini çıkarın, BGR'den RGB kanal sıralamasına dönüştürün, 224x224'e yeniden boyutlandırın ve önceden işleyin
			yuz = video_kare[baslangic2:bitis2, baslangic1:bitis1]
			yuz = cv2.cvtColor(yuz, cv2.COLOR_BGR2RGB)
			yuz = cv2.resize(yuz, (224, 224))
			yuz = img_to_array(yuz)
			yuz = preprocess_input(yuz)

			# yüz ve sınırlayıcı kutuları ilgili listelerine ekleyin
			yuzler.append(yuz)
			yerler.append((baslangic1, baslangic2, bitis1, bitis2))

	# yalnızca en az bir yüz algılandıysa tahminde bulunun
	if len(yuzler) > 0:
		# daha hızlı çıkarım için yukarıdaki "for" döngüsündeki tek tek tahminler yerine tüm yüzlerde aynı anda toplu tahminler yapacağız
		yuzler = np.array(yuzler, dtype="float32")
		tahminler = maskeAgi.predict(yuzler, batch_size=32)

	# yüz konumlarının 2 demetini ve bunlara karşılık gelen konumları döndür
	return (yerler, tahminler)

# yüz dedektör modelimizi diskten yükleyin
txtPath = r"face_detector\deploy.txt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
yuzAgi = cv2.dnn.readNet(txtPath, weightsPath)

# yüz maskesi dedektör modelini diskten yükleyin
maskeModel = load_model("maske_tespiti.model")

# video akışını başlat
print("Video Baslatiliyor")
kamera = VideoStream(src=0).start()

# video akışından kareler üzerinde döngü
while True:
	# akıtılan video akışından çerçeveyi alın ve maksimum 400 piksel genişliğe sahip olacak şekilde yeniden boyutlandırın
	video_kare = kamera.read()
	video_kare = imutils.resize(video_kare, width=400)

	# çerçevedeki yüzleri algılayın ve yüz maskesi takıp takmadıklarını belirleyin
	(yerler, tahminler) = maske_tespiti_ve_tahmini(video_kare, yuzAgi, maskeModel)

	# algılanan yüz konumları ve bunlara karşılık gelen konumlar üzerinde döngü
	for (kutu, tahminler) in zip(yerler, tahminler):
		# sınırlayıcı kutuyu ve tahminleri aç
		(baslangic1, baslangic2, bitis1, bitis2) = kutu
		(mask, withoutMask) = tahminler

		# sınırlayıcı kutuyu ve metni çizmek için kullanacağımız sınıf etiketini ve rengini belirleyin
		if mask > withoutMask:
			etiket = "Maskeli"
		else:
			etiket = "Maskesiz"

		if etiket == "Maskeli":
			renk = (0, 255, 0)
		else:
			renk = (0, 0, 255)

		# etikete olasılığı dahil et
		etiket = "{}: {:.2f}%".format(etiket, max(mask, withoutMask) * 100)

		# etiket ve sınırlayıcı kutu dikdörtgenini çıktı çerçevesinde görüntüle
		cv2.putText(video_kare, etiket, (baslangic1, baslangic2 - 10),
			cv2.FONT_HERSHEY_DUPLEX, 0.45, renk, 2)
		cv2.rectangle(video_kare, (baslangic1, baslangic2), (bitis1, bitis2), renk, 2)

	# çıktı çerçevesini göster
	cv2.imshow("Video Kare", video_kare)
	key = cv2.waitKey(1) & 0xFF

	# 'e' tuşuna basılmışsa, döngüden çık
	if key == ord("e"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
kamera.stop()


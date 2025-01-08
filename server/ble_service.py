import asyncio
import logging
from bleak import BleakClient
from datetime import datetime
from model_inference import DrinkDetectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# UUIDs anpassen
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# MAC-Adresse deiner Smartwatch (Beispiel, bitte anpassen!)
WATCH_MAC_ADDRESS = "54:32:04:22:52:1A"


class BLEService:
    def __init__(self, model_path='drink_detection_model.pkl'):
        self.client = BleakClient(WATCH_MAC_ADDRESS)
        self.model_inference = DrinkDetectionModel(model_path)
        self.window_size = 30

        # Ringpuffer: hier sammeln wir 30 Werte, dann machen wir eine Vorhersage
        self.sample_buffer = []
        self.drink_count = 0  # Anzahl erkannter Trinkvorgänge

        # Offset-Werte (Kalibrierung)
        self.offset_ax = 0.0
        self.offset_ay = 0.0
        self.offset_az = 0.0
        self.offset_gx = 0.0
        self.offset_gy = 0.0
        self.offset_gz = 0.0

        # Callback, den wir später setzen
        self.on_data_callback = None
        self.on_drink_callback = None

    async def connect(self):
        """
        Baut die BLE-Verbindung auf und startet Notification.
        Ruft danach die Kalibrierung auf und startet den Auto-Kalibrierungs-Task.
        """
        logging.info(f"Verbinde mit der Smartwatch: {WATCH_MAC_ADDRESS}")
        await self.client.connect()
        logging.info("Verbindung hergestellt.")

        # Prüfe Service/Char
        services = await self.client.get_services()
        if SERVICE_UUID not in [s.uuid for s in services]:
            logging.error(f"Service UUID {SERVICE_UUID} nicht gefunden.")
            return False
        characteristics = services.get_service(SERVICE_UUID).characteristics
        char_uuids = [c.uuid for c in characteristics]
        if CHARACTERISTIC_UUID not in char_uuids:
            logging.error(f"Characteristic UUID {CHARACTERISTIC_UUID} nicht gefunden.")
            return False

        await self.client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
        logging.info("Benachrichtigungen gestartet. Warte auf Daten...")

        # 1) Einmalige Initialkalibrierung
        await self.calibrate()

        # 2) Regelmäßige Re-Kalibrierung (z. B. alle 2 Minuten)
        #asyncio.create_task(self.auto_calibration_task(interval_seconds=120))

        return True

    async def calibrate(self):
        """
        Sammelt kurz Daten (z. B. 30 Samples) und berechnet deren Mittelwerte
        als Offset. Der Nutzer sollte die Uhr dabei möglichst ruhig halten.
        """
        logging.info("Starte Kalibrierung: Bitte Uhr still halten...")

        calibration_samples = []
        required_samples = 30

        # Temporär Notification-Handler "umleiten" oder wir sammeln passiv?
        # Hier machen wir es aktiv: wir warten, bis wir 30 Datensätze haben.
        while len(calibration_samples) < required_samples:
            data = await self._get_next_sample()
            calibration_samples.append(data)

        # Offsets berechnen
        ax_vals = [s["ax"] for s in calibration_samples]
        ay_vals = [s["ay"] for s in calibration_samples]
        az_vals = [s["az"] for s in calibration_samples]
        gx_vals = [s["gx"] for s in calibration_samples]
        gy_vals = [s["gy"] for s in calibration_samples]
        gz_vals = [s["gz"] for s in calibration_samples]

        self.offset_ax = sum(ax_vals) / required_samples
        self.offset_ay = sum(ay_vals) / required_samples
        self.offset_az = sum(az_vals) / required_samples
        self.offset_gx = sum(gx_vals) / required_samples
        self.offset_gy = sum(gy_vals) / required_samples
        self.offset_gz = sum(gz_vals) / required_samples

        logging.info(
            f"Kalibrierung abgeschlossen. Offsets: "
            f"ax={self.offset_ax:.3f}, ay={self.offset_ay:.3f}, az={self.offset_az:.3f}, "
            f"gx={self.offset_gx:.3f}, gy={self.offset_gy:.3f}, gz={self.offset_gz:.3f}"
        )

    async def auto_calibration_task(self, interval_seconds=120):
        """
        Führt alle X Sekunden eine neue Kalibrierung durch (wenn gewünscht).
        """
        while True:
            await asyncio.sleep(interval_seconds)
            logging.info("Starte automatische Re-Kalibrierung...")
            await self.calibrate()

    async def _get_next_sample(self):
        """
        Wartet auf das nächste Sample aus der Notification-Queue.
        Da wir gerade kein "direktes" Queue-System nutzen, machen wir hier ein
        einfaches Warten über die sample_buffer (Poll).
        """
        while True:
            if self.sample_buffer:
                # Nimm den ersten Datensatz aus dem Buffer
                return self.sample_buffer.pop(0)
            await asyncio.sleep(0.05)

    def notification_handler(self, sender, data):
        """
        Wird aufgerufen, wenn ein neuer BLE-Datenchunk empfangen wird.
        """
        try:
            text = data.decode('utf-8').strip()
            parts = text.split(';')  # Trenner anpassen
            if len(parts) != 8:
                logging.error(f"Unerwartetes Datenformat: {text}")
                return

            # timestamp_str, ax, ay, az, gx, gy, gz, pulse
            # Wir ignorieren pulse in diesem Beispiel
            _, ax, ay, az, gx, gy, gz, _ = parts
            sensor_dict = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ax": float(ax) - self.offset_ax,
                "ay": float(ay) - self.offset_ay,
                "az": float(az) - self.offset_az,
                "gx": float(gx) - self.offset_gx,
                "gy": float(gy) - self.offset_gy,
                "gz": float(gz) - self.offset_gz,
            }

            # Neuen Datensatz in den Buffer legen
            self.sample_buffer.append(sensor_dict)

            # Callback für neu empfangenen Wert (im Frontend anzeigen)
            if self.on_data_callback:
                self.on_data_callback(sensor_dict)

            # Wenn 30 Werte im Buffer, Vorhersage
            if len(self.sample_buffer) >= self.window_size:
                # Nimm genau 30 für das Fenster
                window = self.sample_buffer[:self.window_size]
                # Lösche diese 30 aus dem Buffer
                self.sample_buffer = self.sample_buffer[self.window_size:]

                prediction = self.model_inference.predict(window)
                if prediction == 1:
                    self.drink_count += 1
                    if self.on_drink_callback:
                        self.on_drink_callback(self.drink_count)

        except Exception as e:
            logging.error(f"Fehler beim Verarbeiten der Sensordaten: {e}")

    async def disconnect(self):
        if self.client.is_connected:
            await self.client.stop_notify(CHARACTERISTIC_UUID)
            await self.client.disconnect()
            logging.info("Verbindung zur Smartwatch getrennt.")

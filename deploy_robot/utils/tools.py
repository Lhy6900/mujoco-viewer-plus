import cv2
import zmq
import json
import base64

context = zmq.Context()


def send_img_to_dino(image, item="bottle", modifier="red", HOST="127.0.0.1", PORT="5679"):

    skt = context.socket(zmq.REQ)
    skt.connect("tcp://" + HOST + ":" + PORT)
    try:
        _, img_encoded = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        skt.send(json.dumps([img_base64, item, modifier]).encode("utf-8"))
        received = skt.recv()
        reply = json.loads(received.decode("utf-8"))
        print("Received:", reply)
        return reply
    except Exception as e:
        print(e)
        return []
    finally:
        skt.close()


def mask_frame(frame=None, mask_size=[5, 5], positions=[]):

    if frame is None or frame.shape is None:
        return None
    height, width = frame.shape[:2]
    new_frame = frame.copy()
    half_size = [mask_size[0] // 2, mask_size[1] // 2]

    for pos in positions:
        x, y = pos
        if not (0 <= y < height and 0 <= x < width):
            continue

        top_left = (max(x - half_size[0], 0), max(y - half_size[1], 0))
        bottom_right = (min(x + half_size[0], width - 1), min(y + half_size[1], height - 1))

        cv2.rectangle(new_frame, top_left, bottom_right, (0, 0, 0), thickness=-1)

    return new_frame


def waiting_orders(HOST="0.0.0.0", PORT="5678"):

    try:
        skt = context.socket(zmq.REP)
        skt.bind("tcp://" + HOST + ":" + PORT)
        print("Waiting orders.")
        received = skt.recv()
        data = json.loads(received.decode("utf-8"))
        skt.send(json.dumps("success").encode("utf-8"))
        return data
    except Exception as e:
        print(e)
        return e
    finally:
        skt.close()

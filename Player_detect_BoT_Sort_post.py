from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort, DeepOcSort
from tracklet_merger import Tracklet, TrackletMerger, apply_merge_to_video

INPUT_VIDEO = "test_video/test_5-1.mp4"
OUTPUT_VIDEO = "results/result_tracked_BSort-y12m-1-model-x1-ntt055.mp4"
MODEL_NAME = "models/yolo12m.pt"

TRACKER_TYPE = "botsort"

CONFIDENCE = 0.25
IOU_THRESH = 0.5
MIN_BBOX_HEIGHT = 30

SHOW_PREVIEW = True
TRAIL_LENGTH = 50
BBOX_THICKNESS = 2
FONT_SCALE = 0.6

ENABLE_MERGE = True
MERGED_VIDEO = "results/result_tracked_MERGED.mp4"
MAX_FRAME_GAP = 180
MAX_SPATIAL_DIST = 200
MERGE_COST_THRESH = 0.6

def create_tracker(tracker_type: str):
    if tracker_type == "botsort":
        tracker = BotSort(
            Path("osnet_x1_0_msmt17.pt"),
            "cuda:0",
            False,

            track_high_thresh=0.3,
            track_low_thresh=0.1,
            new_track_thresh=0.55,
            match_thresh=0.82,

            track_buffer=150,

            cmc_method="sof",

            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=True,
        )
    elif tracker_type == "deepocsort":
        tracker = DeepOcSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt"),
            device="cuda:0",
            fp16=False,

            det_thresh=0.3,
            max_age=150,
            min_hits=3,
            iou_threshold=0.3,
            asso_func="giou",

            # Re-ID
            w_association_emb=0.75,
            aw_param=0.5,
        )
    else:
        raise ValueError(f"Неизвестный трекер: {tracker_type}")

    return tracker


def get_color_for_id(track_id: int) -> tuple:
    np.random.seed(int(track_id) * 7)
    return tuple(int(c) for c in np.random.randint(80, 255, size=3))


def draw_tracks(frame, tracked_objects, trails):
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj[:4])
        tid = int(obj[4])
        conf = obj[5]
        color = get_color_for_id(tid)

        cx, cy = (x1 + x2) // 2, y2
        if tid not in trails:
            trails[tid] = []
        trails[tid].append((cx, cy))
        if len(trails[tid]) > TRAIL_LENGTH:
            trails[tid] = trails[tid][-TRAIL_LENGTH:]

        pts = trails[tid]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        label = f"ID {tid} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 2)

    return frame


def main():
    input_path = Path(INPUT_VIDEO)
    if not input_path.exists():
        print(f"[ERROR] Видео не найдено: {input_path.resolve()}")
        return

    # Модель
    print(f"[INFO] Загрузка модели {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # Трекер
    print(f"[INFO] Инициализация трекера: {TRACKER_TYPE}")
    tracker = create_tracker(TRACKER_TYPE)

    # Видео
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть: {input_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(OUTPUT_VIDEO)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    print(f"[INFO] Видео: {w}x{h} @ {fps:.1f} FPS, кадров: {total}")
    print(f"[INFO] Результат: {output_path.resolve()}")
    print(f"[INFO] Обработка...\n")

    trails = {}
    frame_num = 0
    max_simultaneous = 0

    tracklets = {}
    frame_track_data = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model.predict(
            frame,
            conf=CONFIDENCE,
            iou=IOU_THRESH,
            classes=[0],
            verbose=False,
        )

        dets = []
        if results[0].boxes is not None and len(results[0].boxes):
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                bh = xyxy[3] - xyxy[1]
                if bh >= MIN_BBOX_HEIGHT:
                    dets.append([*xyxy, conf, cls])

        if dets:
            dets = np.array(dets, dtype=np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        tracked = tracker.update(dets, frame)

        if len(tracked) > 0:
            frame_track_data[frame_num] = []
            for obj in tracked:
                tid = int(obj[4])
                box = obj[:4].copy()
                conf_val = obj[5]

                frame_track_data[frame_num].append(
                    [*box, tid, conf_val]
                )

                if tid not in tracklets:
                    tracklets[tid] = Tracklet(track_id=tid)
                tracklets[tid].frames.append(frame_num)
                tracklets[tid].boxes.append(box.tolist())

            frame = draw_tracks(frame, tracked, trails)
            current_count = len(tracked)
            max_simultaneous = max(max_simultaneous, current_count)
            count_text = f"On field: {current_count} | Unique IDs: {len(trails)}"
        else:
            count_text = "On field: 0"

        # HUD
        cv2.putText(frame, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame {frame_num}/{total}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Tracker: {TRACKER_TYPE.upper()}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        writer.write(frame)

        if SHOW_PREVIEW:
            preview = cv2.resize(frame, (1280, 720)) if w > 1280 else frame
            cv2.imshow("Football Tracker v2", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Остановлено (q)")
                break

        if frame_num % 100 == 0:
            pct = frame_num / total * 100 if total > 0 else 0
            n = len(tracked) if len(tracked) > 0 else 0
            print(f"  Frame {frame_num}/{total} ({pct:.1f}%) | "
                  f"Tracked: {n} | Unique IDs: {len(trails)}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"[DONE] Обработано кадров: {frame_num}")
    print(f"[DONE] Всего уникальных ID: {len(trails)}")
    print(f"[DONE] Макс. одновременно на поле: {max_simultaneous}")
    print(f"[DONE] Результат: {output_path.resolve()}")

    if ENABLE_MERGE and len(tracklets) > 0:
        print(f"\n{'='*50}")
        print(f"ПОСТ-ОБРАБОТКА ТРЕКОВ")
        print(f"{'='*50}")

        merger = TrackletMerger(
            max_frame_gap=MAX_FRAME_GAP,
            max_spatial_dist=MAX_SPATIAL_DIST,
            merge_cost_thresh=MERGE_COST_THRESH,
        )

        id_map = merger.merge(tracklets)

        unique_before = len(set(tracklets.keys()))
        unique_after = len(set(id_map.values()))
        print(f"\n[MERGE] ID до объединения: {unique_before}")
        print(f"[MERGE] ID после объединения: {unique_after}")
        print(f"[MERGE] Сокращение: {unique_before - unique_after} треков")

        merges = [(old, new) for old, new in id_map.items() if old != new]
        if merges:
            print(f"\n[MERGE] Объединения:")
            grouped = defaultdict(list)
            for old, new in merges:
                grouped[new].append(old)
            for new_id, old_ids in grouped.items():
                all_ids = sorted([new_id] + old_ids)
                durations = []
                for tid in all_ids:
                    if tid in tracklets:
                        t = tracklets[tid]
                        durations.append(
                            f"ID {tid} (frames {t.start_frame}-{t.end_frame})"
                        )
                print(f"  → ID {new_id}: {' + '.join(durations)}")

        merged_path = Path(MERGED_VIDEO)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[MERGE] Создание видео с объединёнными ID")
        apply_merge_to_video(
            INPUT_VIDEO, str(merged_path),
            frame_track_data, id_map, fps
        )
        print(f"[MERGE] Готово: {merged_path.resolve()}")


if __name__ == "__main__":
    main()
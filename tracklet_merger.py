import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Tracklet:
    track_id: int
    frames: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)

    @property
    def start_frame(self):
        return self.frames[0] if self.frames else 0

    @property
    def end_frame(self):
        return self.frames[-1] if self.frames else 0

    @property
    def duration(self):
        return self.end_frame - self.start_frame + 1

    @property
    def start_pos(self):
        if not self.boxes:
            return np.zeros(2)
        b = self.boxes[0]
        return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

    @property
    def end_pos(self):
        if not self.boxes:
            return np.zeros(2)
        b = self.boxes[-1]
        return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

    @property
    def end_velocity(self):
        if len(self.boxes) < 3:
            return np.zeros(2)
        positions = []
        for b in self.boxes[-5:]:
            positions.append(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]))
        velocities = [positions[i+1] - positions[i]
                      for i in range(len(positions)-1)]
        return np.mean(velocities, axis=0)

    @property
    def start_velocity(self):
        if len(self.boxes) < 3:
            return np.zeros(2)
        positions = []
        for b in self.boxes[:5]:
            positions.append(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]))
        velocities = [positions[i+1] - positions[i]
                      for i in range(len(positions)-1)]
        return np.mean(velocities, axis=0)

    @property
    def avg_size(self):
        if not self.boxes:
            return np.zeros(2)
        sizes = [(b[2]-b[0], b[3]-b[1]) for b in self.boxes]
        return np.mean(sizes, axis=0)

    @property
    def mean_embedding(self):
        if not self.embeddings:
            return None
        valid = [e for e in self.embeddings if e is not None]
        if not valid:
            return None
        return np.mean(valid, axis=0)


class TrackletMerger:
    def __init__(self,
                 max_frame_gap=180,
                 max_spatial_dist=200,
                 min_tracklet_len=5,
                 embedding_thresh=0.4,
                 velocity_weight=0.5,
                 size_thresh=0.5,
                 merge_cost_thresh=0.6):
        self.max_frame_gap = max_frame_gap
        self.max_spatial_dist = max_spatial_dist
        self.min_tracklet_len = min_tracklet_len
        self.embedding_thresh = embedding_thresh
        self.velocity_weight = velocity_weight
        self.size_thresh = size_thresh
        self.merge_cost_thresh = merge_cost_thresh

    def merge(self, tracklets: dict) -> dict:
        valid = {tid: t for tid, t in tracklets.items()
                 if len(t.frames) >= self.min_tracklet_len}

        short = {tid: t for tid, t in tracklets.items()
                 if len(t.frames) < self.min_tracklet_len}

        print(f"[MERGE] Всего треков: {len(tracklets)}")
        print(f"[MERGE] Валидных (>={self.min_tracklet_len} кадров): {len(valid)}")
        print(f"[MERGE] Коротких (отброшены): {len(short)}")

        sorted_ids = sorted(valid.keys(), key=lambda tid: valid[tid].start_frame)

        id_map = {tid: tid for tid in tracklets}
        merge_count = 0

        merged_into = set()

        for i, tid_a in enumerate(sorted_ids):
            if tid_a in merged_into:
                continue

            t_a = valid[tid_a]
            best_match = None
            best_cost = self.merge_cost_thresh

            for j in range(i + 1, len(sorted_ids)):
                tid_b = sorted_ids[j]
                if tid_b in merged_into:
                    continue

                t_b = valid[tid_b]

                frame_gap = t_b.start_frame - t_a.end_frame
                if frame_gap < 0:
                    continue
                if frame_gap > self.max_frame_gap:
                    break

                cost = self._compute_merge_cost(t_a, t_b, frame_gap)
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_match = tid_b

            if best_match is not None:
                id_map[best_match] = self._resolve_id(id_map, tid_a)
                merged_into.add(best_match)
                merge_count += 1

                t_b = valid[best_match]
                t_a.frames.extend(t_b.frames)
                t_a.boxes.extend(t_b.boxes)
                t_a.embeddings.extend(t_b.embeddings)

        for tid_s, t_s in short.items():
            best_match = None
            best_cost = self.merge_cost_thresh

            for tid_v in sorted_ids:
                if tid_v in merged_into:
                    continue
                t_v = valid[tid_v]

                gap_after = t_s.start_frame - t_v.end_frame
                gap_before = t_v.start_frame - t_s.end_frame

                if 0 < gap_after <= self.max_frame_gap:
                    cost = self._compute_merge_cost(t_v, t_s, gap_after)
                    if cost is not None and cost < best_cost:
                        best_cost = cost
                        best_match = tid_v
                elif 0 < gap_before <= self.max_frame_gap:
                    cost = self._compute_merge_cost(t_s, t_v, gap_before)
                    if cost is not None and cost < best_cost:
                        best_cost = cost
                        best_match = tid_v

            if best_match is not None:
                id_map[tid_s] = self._resolve_id(id_map, best_match)
                merge_count += 1

        for tid in id_map:
            id_map[tid] = self._resolve_id(id_map, tid)

        unique_after = len(set(id_map.values()))
        print(f"[MERGE] Объединений: {merge_count}")
        print(f"[MERGE] Уникальных ID после: {unique_after}")

        return id_map

    def _compute_merge_cost(self, t_a: Tracklet, t_b: Tracklet,
                             frame_gap: int) -> float:
        costs = []
        weights = []

        predicted_pos = t_a.end_pos + t_a.end_velocity * frame_gap
        actual_pos = t_b.start_pos

        spatial_dist = np.linalg.norm(predicted_pos - actual_pos)
        if spatial_dist > self.max_spatial_dist:
            return None

        spatial_cost = spatial_dist / self.max_spatial_dist
        costs.append(spatial_cost)
        weights.append(0.35)

        direct_dist = np.linalg.norm(t_a.end_pos - t_b.start_pos)
        if direct_dist > self.max_spatial_dist * 1.5:
            return None

        direct_cost = direct_dist / (self.max_spatial_dist * 1.5)
        costs.append(direct_cost)
        weights.append(0.15)

        size_a = t_a.avg_size
        size_b = t_b.avg_size
        if size_a[0] > 0 and size_b[0] > 0:
            size_diff = np.abs(size_a - size_b) / (size_a + 1e-6)
            if np.any(size_diff > self.size_thresh):
                return None
            size_cost = np.mean(size_diff) / self.size_thresh
            costs.append(size_cost)
            weights.append(0.1)

        vel_a = t_a.end_velocity
        vel_b = t_b.start_velocity
        if np.linalg.norm(vel_a) > 0.5 and np.linalg.norm(vel_b) > 0.5:
            vel_cos = np.dot(vel_a, vel_b) / (
                np.linalg.norm(vel_a) * np.linalg.norm(vel_b) + 1e-6)
            vel_cost = (1 - vel_cos) / 2  # [0, 1]
            costs.append(vel_cost)
            weights.append(0.15)

        emb_a = t_a.mean_embedding
        emb_b = t_b.mean_embedding
        if emb_a is not None and emb_b is not None:
            cos_sim = np.dot(emb_a, emb_b) / (
                np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-6)
            emb_cost = (1 - cos_sim) / 2
            if emb_cost > self.embedding_thresh:
                return None
            costs.append(emb_cost)
            weights.append(0.25)

        gap_cost = frame_gap / self.max_frame_gap
        costs.append(gap_cost)
        weights.append(0.1 if emb_a is not None else 0.2)

        weights = np.array(weights)
        weights /= weights.sum()
        total_cost = np.dot(costs, weights)

        return total_cost

    def _resolve_id(self, id_map, tid):
        visited = set()
        current = tid
        while id_map.get(current, current) != current:
            if current in visited:
                break
            visited.add(current)
            current = id_map[current]
        return current


def apply_merge_to_video(input_video, output_video, track_data, id_map,
                          fps=30.0):
    import cv2

    cap = cv2.VideoCapture(str(input_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(output_video),
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))

    frame_num = 0
    trails = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        detections = track_data.get(frame_num, [])

        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            old_id = int(det[4])
            conf = det[5]

            new_id = id_map.get(old_id, old_id)

            np.random.seed(int(new_id) * 7)
            color = tuple(int(c) for c in np.random.randint(80, 255, 3))

            cx, cy = (x1 + x2) // 2, y2
            if new_id not in trails:
                trails[new_id] = []
            trails[new_id].append((cx, cy))
            if len(trails[new_id]) > 50:
                trails[new_id] = trails[new_id][-50:]

            for i in range(1, len(trails[new_id])):
                alpha = i / len(trails[new_id])
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, trails[new_id][i-1], trails[new_id][i],
                         color, thickness)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {new_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, label, (x1+3, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        active = len(detections)
        unique = len(set(id_map.get(int(d[4]), int(d[4]))
                         for d in detections)) if detections else 0
        cv2.putText(frame, f"On field: {active} | Unique: {len(trails)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_num} [MERGED]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[MERGE] Видео сохранено: {output_video}")
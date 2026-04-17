# ACAS v2 Architecture Overhaul: Intelligent Face-Centric PTZ with MOT + RE-ID
 
## Problem Statement
 
The current pipeline runs YOLOv8-Pose as the tracking backbone with face recognition as a secondary layer. This causes:
 
1. **Duplicate recognitions** — pose tracks and face identities diverge; the same person gets counted multiple times or flips between known/unknown.
2. **Identity fragmentation** — when a face becomes temporarily invisible (head turned, looking down), the pose track continues but identity is lost, creating a new "unknown" sighting.
3. **Wasted compute** — full pose estimation (17 keypoints) is expensive and unused for attendance; only bounding boxes matter.
4. **No quality gating** — the pipeline attempts recognition on every detected face regardless of angle, blur, or resolution, leading to low-confidence matches and false positives.
 
## Design Philosophy
 
**"Face is the only source of truth for identity. Everything else is a bridge."**
 
The new architecture treats face recognition as the **identity oracle** and builds a robust **tracking + RE-ID scaffold** around it to maintain identity continuity when faces are temporarily unavailable. The PTZ system becomes **face-quality-aware**, actively seeking the best possible face captures rather than just following person blobs.
 
---
 
## Architecture Overview (New)
 
```
Frame from RTSP
  │
  ├─► [GPU Batch] YOLOv8l Person Detection (bbox only, no pose)
  │     └─► person_detections[] with confidence
  │
  ├─► [GPU Batch] SCRFD Face Detection
  │     └─► face_detections[] with landmarks + bbox
  │
  ├─► Face-to-Person Spatial Association
  │     └─► Each face matched to its enclosing person bbox
  │     └─► Unmatched faces (no person bbox) → still processed
  │     └─► Unmatched persons (no face) → RE-ID candidates
  │
  ├─► BoT-SORT MOT Tracker (on person bboxes)
  │     └─► Assigns persistent track_id to each person across frames
  │     └─► Kalman filter prediction + IoU + optional RE-ID appearance cost
  │     └─► Camera Motion Compensation (critical for PTZ — camera moves!)
  │
  ├─► Face Quality Gate (per detected face)
  │     ├─► Compute quality score: face size, yaw/pitch from landmarks,
  │     │   blur (Laplacian variance), occlusion ratio
  │     ├─► If quality >= RECOGNITION_THRESHOLD → full pipeline:
  │     │     ├─► Liveness Check (MiniFASNet)
  │     │     ├─► ArcFace Embedding → FAISS Lookup
  │     │     └─► Result: identity OR "unknown with embedding stored"
  │     ├─► If quality < RECOGNITION_THRESHOLD but >= TRACKING_THRESHOLD:
  │     │     └─► Use face bbox for tracking association only (skip recognition)
  │     └─► If quality < TRACKING_THRESHOLD:
  │           └─► Discard (too degraded to be useful)
  │
  ├─► Identity-Track Binding
  │     └─► When face recognized → bind identity to MOT track_id
  │     └─► track_id now carries that identity in subsequent frames
  │     └─► Identity stored in TrackState with confidence + timestamp
  │
  ├─► RE-ID Bridge (for tracks that lose face visibility)
  │     ├─► OSNet x1_0 appearance embedding on person crop
  │     ├─► Match against gallery of recently-recognized track appearances
  │     └─► If match: inherit identity (marked as reid_bridged)
  │     └─► If no match: remain unidentified (do NOT duplicate as unknown)
  │
  ├─► Best-Shot Gallery (per identity per session)
  │     └─► Keep top-K face crops ranked by quality score
  │     └─► Used for: dashboard thumbnails, re-enrollment, audit
  │     └─► Updated only when a new crop beats the lowest in gallery
  │
  └─► Output: unified TrackedPerson[] with:
        - track_id (MOT), identity (face), confidence, tracking_method
        - face_quality_score, best_shot_url
        - activity_label (future: placeholder for head pose / micro-action)
```
 
---
 
## Detailed Component Specifications
 
### 1. MOT Tracker: BoT-SORT with Camera Motion Compensation
 
**Why BoT-SORT, not simple IoU/SORT:**
PTZ cameras move. When the camera pans, ALL bounding boxes shift simultaneously. Simple IoU tracking breaks completely because predicted positions (from Kalman filter) don't account for camera motion. BoT-SORT solves this with Global Motion Compensation (GMC) using sparse optical flow or affine estimation between frames.
 
**Implementation:**
 
Create `backend/app/services/mot_tracker.py`:
 
```python
class BoTSORTTracker:
    """
    Multi-Object Tracker with camera motion compensation.
    Tracks person bounding boxes across frames with stable track_ids.
    Integrates optional RE-ID appearance cost for association.
    """
    
    # Core components:
    # - Kalman Filter with (x, y, w, h, vx, vy, vw, vh) state vector
    # - Camera Motion Compensation via sparse optical flow (cv2.calcOpticalFlowPyrLK)
    #   or affine estimation (cv2.estimateAffine2D)
    # - Two-stage association (ByteTrack-style):
    #   Stage 1: High-confidence detections matched to tracks via IoU + appearance
    #   Stage 2: Low-confidence detections matched to remaining tracks via IoU only
    # - Track lifecycle: TENTATIVE (2 frames) → CONFIRMED → LOST (max_age frames) → DELETED
    
    def __init__(self, config: BoTSORTConfig):
        self.max_age = config.track_buffer  # frames before deleting lost track
        self.min_hits = config.min_hits  # frames before confirming tentative
        self.high_thresh = config.track_high_thresh  # 0.5
        self.low_thresh = config.track_low_thresh  # 0.1
        self.match_thresh = config.match_thresh  # 0.8
        self.gmc = GlobalMotionCompensation(method=config.gmc_method)  # sparseOptFlow
        self.reid_weight = config.appearance_weight  # 0.0 to 0.3
    
    def update(self, detections, frame, reid_embeddings=None) -> list[Track]:
        """
        Args:
            detections: list of (bbox, confidence, class_id)
            frame: current BGR frame (for GMC)
            reid_embeddings: optional appearance embeddings per detection
        Returns:
            list of active Track objects with track_id, bbox, state
        """
        # 1. Estimate camera motion, warp predicted track positions
        # 2. Predict new positions via Kalman filter
        # 3. Split detections into high/low confidence
        # 4. Stage 1: match high-conf to tracks (IoU + appearance cost)
        # 5. Stage 2: match low-conf to unmatched tracks (IoU only)
        # 6. Initialize new tracks from unmatched high-conf detections
        # 7. Update track states, remove stale
        ...
    
    def camera_motion_compensate(self, frame):
        """Critical for PTZ: estimate affine transform from frame-to-frame."""
        ...
```
 
**Key config:**
```bash
# BoT-SORT Tracker Config
MOT_TRACK_HIGH_THRESH=0.5
MOT_TRACK_LOW_THRESH=0.1
MOT_MATCH_THRESH=0.8
MOT_TRACK_BUFFER=30          # frames to keep lost track alive (~1 second at 30fps)
MOT_MIN_HITS=3               # frames before track is confirmed
MOT_GMC_METHOD=sparseOptFlow  # Global Motion Compensation method
MOT_APPEARANCE_WEIGHT=0.2    # Weight for RE-ID cost in association (0=disabled)
```
 
**Why this matters for your PTZ system:**
- When PTZ pans to a new area, GMC prevents all tracks from being lost and recreated
- When PTZ zooms, bbox sizes change dramatically — Kalman filter with proper state handles this
- Two-stage association (ByteTrack) keeps partially-occluded persons tracked even at low confidence
- Track lifecycle prevents premature identity loss during brief occlusions
 
---
 
### 2. Face Quality Gate
 
**Why:** Currently every detected face goes through the full ArcFace + FAISS pipeline. A blurry, side-profile, 20×20 pixel face will produce a poor embedding that either (a) matches the wrong person or (b) creates a false "unknown." Quality gating prevents this.
 
Create `backend/app/services/face_quality.py`:
 
```python
class FaceQualityAssessor:
    """
    Fast face quality scoring using geometric + image cues.
    No additional neural network needed — derived from SCRFD landmarks
    and basic image statistics.
    """
    
    def assess(self, face_crop: np.ndarray, landmarks: np.ndarray) -> FaceQualityScore:
        """
        Returns composite score 0.0-1.0 and individual component scores.
        
        Components (weighted):
        - face_size: 0.20 — pixel area of face bbox (min 40x40 for recognition)
        - yaw_angle: 0.25 — estimated from landmark asymmetry (nose-to-eye ratios)
        - pitch_angle: 0.15 — estimated from nose-to-mouth vs nose-to-eye distances
        - blur_score: 0.20 — Laplacian variance of the face crop
        - illumination: 0.10 — histogram spread / mean brightness
        - occlusion: 0.10 — landmark confidence scores from SCRFD
        """
        
        # Yaw estimation from 5-point landmarks (SCRFD output):
        # left_eye, right_eye, nose, left_mouth, right_mouth
        # yaw ≈ atan2(nose_x - midpoint_eyes_x, inter_eye_distance)
        # Simple but effective for gating purposes
        
        # Pitch estimation:
        # Compare nose-to-eye distance vs nose-to-mouth distance
        # Looking up: nose closer to eyes; Looking down: nose closer to mouth
        
        # Blur via Laplacian:
        # cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        # Low variance = blurry
        
        return FaceQualityScore(
            composite=weighted_sum,
            face_size=size_score,
            yaw=yaw_score,
            pitch=pitch_score,
            blur=blur_score,
            illumination=illum_score,
            occlusion=occl_score,
            estimated_yaw_degrees=yaw_deg,
            estimated_pitch_degrees=pitch_deg
        )
 
# Quality thresholds (configurable):
QUALITY_RECOGNITION_THRESH = 0.55   # Above: full ArcFace recognition
QUALITY_TRACKING_THRESH = 0.25      # Above: use for face-track association
QUALITY_DISCARD_THRESH = 0.25       # Below: ignore face completely
```
 
**Impact on PTZ brain:**
The PTZ brain now receives quality scores and can make intelligent decisions:
- "I see person X at low face quality (side profile) → zoom in and wait for them to turn"
- "Person X has quality 0.8 → good shot captured, move to next unrecognized person"
- "No faces above recognition threshold in this preset → move on faster"
 
---
 
### 3. RE-ID with OSNet (Proper Model, Not Hacky Histograms)
 
**Why OSNet over color histograms:**
The previous prompt suggested color histograms as "Option A." This is fragile — lighting changes, similar clothing between people, and PTZ zoom level changes will destroy histogram-based matching. OSNet x1_0 is only ~8MB ONNX, runs in <2ms on GPU, and provides robust 512-d appearance embeddings specifically designed for person re-identification. The marginal GPU cost is negligible compared to the accuracy gain.
 
Create `backend/app/services/reid_engine.py`:
 
```python
class REIDEngine:
    """
    Person RE-ID using OSNet x1_0 ONNX model.
    Provides 512-d appearance embeddings for person crops.
    """
    
    def __init__(self, model_path: str = "models/osnet_x1_0.onnx"):
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.input_size = (256, 128)  # H x W — standard RE-ID input
        
    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """Extract 512-d appearance embedding from person crop."""
        # Resize, normalize (ImageNet mean/std), CHW, batch
        blob = self._preprocess(person_crop)
        embedding = self.session.run(None, {self.input_name: blob})[0]
        return embedding / np.linalg.norm(embedding)  # L2 normalize
    
    def batch_extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Batch extraction for efficiency."""
        ...
 
 
class REIDGallery:
    """
    Maintains short-lived appearance gallery for identity bridging.
    
    Lifecycle:
    1. Person recognized by face → store their person crop embedding as anchor
    2. Face lost but person bbox persists → match crop against gallery
    3. Face reappears → confirm or correct the bridged identity
    4. Anchors expire after TTL or preset change
    """
    
    def __init__(self, config: REIDConfig):
        self.anchors: dict[str, REIDAnchor] = {}  # identity_id → anchor
        self.pending_bridges: dict[int, list[BridgedCycle]] = {}  # track_id → cycles
        self.ttl = config.anchor_ttl_seconds  # 30s default
        self.match_threshold = config.match_threshold  # 0.45 for OSNet (cosine)
        self.preset_scope = config.preset_scoped  # True — invalidate on preset change
    
    def register_anchor(self, identity_id: str, track_id: int, 
                         person_crop: np.ndarray, preset_id: str):
        """Called when face recognition succeeds — store appearance anchor."""
        embedding = self.reid_engine.extract(person_crop)
        self.anchors[identity_id] = REIDAnchor(
            embedding=embedding,
            track_id=track_id,
            preset_id=preset_id,
            registered_at=time.time(),
            # Keep multiple embeddings for robustness (gallery of 3-5)
            embedding_gallery=[embedding]
        )
    
    def match(self, person_crop: np.ndarray, preset_id: str) -> Optional[REIDMatch]:
        """
        Match a faceless person crop against the gallery.
        Returns best match above threshold, or None.
        """
        query_emb = self.reid_engine.extract(person_crop)
        best_match = None
        best_score = 0.0
        
        for identity_id, anchor in self.anchors.items():
            if self.preset_scope and anchor.preset_id != preset_id:
                continue
            if time.time() - anchor.registered_at > self.ttl:
                continue
            
            # Compare against gallery of embeddings (max similarity)
            score = max(cosine_similarity(query_emb, e) for e in anchor.embedding_gallery)
            if score > self.match_threshold and score > best_score:
                best_score = score
                best_match = REIDMatch(identity_id=identity_id, confidence=score)
        
        return best_match
    
    def bridge_cycle(self, track_id: int, cycle_id: int, 
                      reid_embedding: np.ndarray, matched_identity: str):
        """Record a cycle where identity was inferred via RE-ID."""
        ...
    
    def resolve_bridges(self, track_id: int, confirmed_identity: str) -> list[int]:
        """
        When face reappears and confirms identity, retroactively assign
        all bridged cycles to that identity.
        Returns list of cycle_ids that were resolved.
        """
        ...
    
    def on_preset_change(self, old_preset: str, new_preset: str):
        """Invalidate or deprioritize anchors from old preset."""
        ...
```
 
**Config:**
```bash
# RE-ID Engine
REID_MODEL_PATH=models/osnet_x1_0.onnx
REID_MATCH_THRESHOLD=0.45        # Cosine similarity (OSNet-specific; lower than ArcFace)
REID_ANCHOR_TTL_SECONDS=30       # Anchor expiry
REID_ANCHOR_GALLERY_SIZE=5       # Number of appearance samples per anchor
REID_PRESET_SCOPED=true          # Invalidate anchors on PTZ preset change
REID_MAX_BRIDGE_SECONDS=60       # Max continuous RE-ID bridging before forcing sighting end
```
 
---
 
### 4. Intelligent PTZ Brain (Face-Quality-Aware)
 
**The current PTZ brain** cycles through presets and tracks persons by position. The new brain becomes **face-quality-aware** and **identity-priority-aware**.
 
Modify `backend/app/services/ptz_brain.py`:
 
**New state machine:**
 
```
SCANNING
  │ Person detected, no face or low quality face
  ▼
APPROACHING
  │ PTZ adjusting to get better face angle/zoom
  │ Using person bbox center + predicted face position
  ▼
CAPTURING
  │ Face quality above recognition threshold
  │ Running ArcFace recognition
  │ Collecting best-shot candidates
  ▼
IDENTIFIED (or UNKNOWN_CAPTURED)
  │ Identity confirmed (or unknown with high-quality embedding stored)
  │ Decision: stay on this person or move to next
  ▼
DWELLING
  │ Monitoring identified person for attendance dwell time
  │ RE-ID maintaining identity if face lost temporarily
  │ Will move on when: dwell target met / new unrecognized person detected
  ▼
RETURNING
  │ Moving back to preset patrol route
  │ Preset change triggers RE-ID gallery cleanup
  ▼
SCANNING (loop)
```
 
**Key intelligence behaviors:**
 
```python
class IntelligentPTZBrain:
    """
    Face-quality-aware PTZ controller.
    
    Priority system:
    1. UNRECOGNIZED persons (need initial face capture) — highest
    2. RECOGNIZED persons below dwell threshold (need more time)  
    3. RE-ID bridged persons (face lost, might need reconfirmation)
    4. Patrol scan (no persons of interest) — lowest
    
    Quality-seeking behaviors:
    - When face detected but quality too low: micro-adjust PTZ
      (zoom in slightly, hold position for next frame)
    - When person detected but no face: predict face position
      from person bbox top (face is usually top 25% of person bbox)
    - When multiple unrecognized persons: prioritize by face
      quality potential (larger bbox = probably closer = better chance)
    """
    
    def compute_face_seeking_adjustment(self, person_bbox, face_quality):
        """
        Given current person position and face quality, compute PTZ adjustment
        to improve face capture quality.
        """
        if face_quality is None:
            # No face detected — zoom toward upper portion of person bbox
            target_y = person_bbox.top + person_bbox.height * 0.15
            target_x = person_bbox.center_x
            zoom_adjust = +1  # zoom in to get face resolution
        elif face_quality.estimated_yaw_degrees > 45:
            # Side profile — hold position and wait (person may turn)
            # Or: if multiple presets cover this area, switch to 
            # the preset with better face angle
            zoom_adjust = 0
        elif face_quality.face_size < 80:
            # Face too small — zoom in
            zoom_adjust = +2
        elif face_quality.blur < 0.3:
            # Motion blur — reduce zoom to widen DOF, or wait
            zoom_adjust = -1
        else:
            # Good quality — hold steady for capture
            zoom_adjust = 0
        
        return PTZAdjustment(pan, tilt, zoom_adjust)
    
    def select_next_target(self, tracked_persons, current_target):
        """
        Smart target selection with priority scoring.
        """
        candidates = []
        for person in tracked_persons:
            score = 0.0
            
            if not person.identity:
                score += 100  # Unrecognized = highest priority
                # Bonus for persons with higher face quality potential
                score += person.face_quality.composite * 20 if person.face_quality else 0
            elif person.tracking_method == 'reid_bridge':
                score += 50  # RE-ID bridged = needs reconfirmation
                score += (time.time() - person.last_face_seen) * 2  # urgency increases
            elif person.dwell_time < person.required_dwell:
                score += 30  # Still needs dwell time
            
            # Penalize far-from-current-position targets (minimize PTZ movement)
            score -= distance_to_current(person.position) * 0.5
            
            candidates.append((person, score))
        
        return max(candidates, key=lambda x: x[1])[0] if candidates else None
```
 
---
 
### 5. Best-Shot Gallery System
 
Create `backend/app/services/best_shot.py`:
 
```python
class BestShotGallery:
    """
    Maintains top-K face crops per identity per session.
    Used for: dashboard display, re-enrollment quality, audit trail.
    
    Only updates when a new capture beats the lowest-quality in gallery.
    Stored in MinIO with metadata in PostgreSQL.
    """
    
    def __init__(self, gallery_size: int = 5):
        self.gallery_size = gallery_size
    
    async def maybe_update(self, identity_id: str, session_id: str,
                            face_crop: np.ndarray, quality: FaceQualityScore):
        """
        Check if this face crop is better than worst in gallery.
        If yes, replace it. If gallery not full, always add.
        """
        ...
    
    async def get_best_shot(self, identity_id: str, session_id: str) -> Optional[bytes]:
        """Return highest-quality face crop for this identity in this session."""
        ...
```
 
---
 
### 6. Identity-Track Binding State
 
Create `backend/app/services/identity_state.py`:
 
```python
@dataclass
class TrackIdentityState:
    """
    The binding between a MOT track and a face identity.
    This is the core data structure that prevents duplicates.
    """
    track_id: int                    # From BoT-SORT
    identity_id: Optional[str]       # From face recognition (None = unidentified)
    identity_confidence: float       # ArcFace similarity score
    tracking_method: str             # 'face' | 'reid_bridge' | 'face_reconfirmed'
    
    # Face tracking
    last_face_seen_at: float         # timestamp
    last_face_quality: float         # quality score when last seen
    face_seen_count: int             # total frames with face visible
    
    # RE-ID bridging
    reid_bridge_start: Optional[float]
    reid_bridge_confidence: Optional[float]
    reid_bridged_cycles: int
    
    # Sighting continuity
    first_seen_at: float
    last_seen_at: float
    preset_id: str
    
    # Future: activity state
    activity_label: Optional[str]    # placeholder for head pose, attention, etc.
    activity_confidence: Optional[float]
 
 
class IdentityTrackManager:
    """
    Manages the mapping between MOT tracks and face identities.
    
    RULES:
    1. One identity per track. Once bound, only face re-recognition can change it.
    2. One track per identity per preset. If same face appears in two bboxes,
       it's a detection error — keep the higher-confidence one.
    3. RE-ID can bridge but never create or change identities.
    4. Unknown tracks with same face embedding don't create duplicate unknowns —
       they share a single "unknown_<hash>" provisional identity.
    """
    
    def bind_identity(self, track_id: int, identity_id: str, 
                       confidence: float, method: str = 'face'):
        """Bind an identity to a track. Highest confidence wins."""
        ...
    
    def get_identity_for_track(self, track_id: int) -> Optional[TrackIdentityState]:
        ...
    
    def get_track_for_identity(self, identity_id: str, preset_id: str) -> Optional[int]:
        """Prevent duplicate tracks for same identity in same preset."""
        ...
    
    def on_track_lost(self, track_id: int):
        """Track deleted by MOT — finalize sighting."""
        ...
    
    def on_preset_change(self, old_preset: str, new_preset: str):
        """PTZ moved — transition all tracks appropriately."""
        ...
```
 
---
 
### 7. Updated Pipeline Orchestrator
 
Modify `backend/app/services/ai_pipeline.py`:
 
```python
class IntelligentPipeline:
    """
    Orchestrates the full frame processing pipeline.
    
    Per-frame flow:
    1. Person detection (YOLOv8l) + Face detection (SCRFD) — can run in parallel on GPU
    2. Face-to-person association (spatial)
    3. MOT tracker update (BoT-SORT on person bboxes)
    4. Face quality assessment
    5. For high-quality faces: liveness → ArcFace → FAISS → identity binding
    6. For tracks without face: RE-ID matching
    7. PTZ brain decision
    8. Best-shot gallery update
    9. Sighting engine update
    """
    
    async def process_frame(self, frame: np.ndarray, camera_id: str, 
                             preset_id: str) -> FrameResult:
        
        # Step 1: Parallel detection
        person_dets, face_dets = await asyncio.gather(
            self.person_detector.detect(frame),  # YOLOv8l bbox-only
            self.face_detector.detect(frame)       # SCRFD
        )
        
        # Step 2: Spatial association
        associations = self.associate_faces_to_persons(person_dets, face_dets)
        # Returns: list of (person_det, Optional[face_det])
        
        # Step 3: MOT tracking with optional RE-ID appearance cost
        reid_embs = None
        if self.config.mot_appearance_weight > 0:
            reid_embs = self.reid_engine.batch_extract(
                [self._crop(frame, p.bbox) for p, _ in associations]
            )
        
        tracks = self.mot_tracker.update(
            detections=[(p.bbox, p.conf, 0) for p, _ in associations],
            frame=frame,
            reid_embeddings=reid_embs
        )
        
        # Step 4-6: Per-track processing
        results = []
        for track in tracks:
            assoc = self._find_association_for_track(track, associations)
            person_det, face_det = assoc
            
            if face_det is not None:
                # Step 4: Quality gate
                quality = self.face_quality.assess(
                    self._crop(frame, face_det.bbox), face_det.landmarks
                )
                
                if quality.composite >= self.config.recognition_threshold:
                    # Step 5: Full recognition pipeline
                    if self.liveness.check(self._crop(frame, face_det.bbox)):
                        embedding = self.arcface.extract(
                            self._align_face(frame, face_det.landmarks)
                        )
                        identity, confidence = self.face_repo.search(embedding)
                        
                        if identity:
                            self.identity_manager.bind_identity(
                                track.id, identity, confidence, method='face'
                            )
                            # Register RE-ID anchor for this identity
                            self.reid_gallery.register_anchor(
                                identity, track.id,
                                self._crop(frame, person_det.bbox), preset_id
                            )
                            # Update best-shot gallery
                            await self.best_shot.maybe_update(
                                identity, self.session_id,
                                self._crop(frame, face_det.bbox), quality
                            )
                        else:
                            # Unknown — store embedding for future matching
                            # Use hash-based provisional ID to prevent duplicates
                            prov_id = f"unknown_{hash(embedding.tobytes())[:8]}"
                            self.identity_manager.bind_identity(
                                track.id, prov_id, 0.0, method='face'
                            )
            else:
                # Step 6: No face visible — attempt RE-ID bridge
                state = self.identity_manager.get_identity_for_track(track.id)
                if state and state.identity_id:
                    # Already identified in a previous frame — extend via RE-ID
                    reid_match = self.reid_gallery.match(
                        self._crop(frame, track.bbox), preset_id
                    )
                    if reid_match and reid_match.identity_id == state.identity_id:
                        # Confirmed — same person, just face not visible
                        state.tracking_method = 'reid_bridge'
                        state.reid_bridged_cycles += 1
                elif not state or not state.identity_id:
                    # Never identified — try RE-ID against all anchors
                    reid_match = self.reid_gallery.match(
                        self._crop(frame, track.bbox), preset_id
                    )
                    if reid_match:
                        self.identity_manager.bind_identity(
                            track.id, reid_match.identity_id,
                            reid_match.confidence, method='reid_bridge'
                        )
            
            results.append(self._build_tracked_person(track, quality, state))
        
        # Step 7: PTZ brain decision
        ptz_action = self.ptz_brain.decide(results, preset_id)
        
        # Step 8-9: Sighting engine
        await self.sighting_engine.update(results, camera_id, preset_id)
        
        return FrameResult(tracked_persons=results, ptz_action=ptz_action)
```
 
---
 
## Model Changes
 
| Current | New | Size | Purpose |
|---------|-----|------|---------|
| `yolov8x-pose.onnx` | **REMOVE** | ~260MB | — |
| — | `yolov8l.onnx` | ~130MB | Person detection (bbox only) |
| `buffalo_l/det_10g.onnx` | **KEEP** | ~16MB | SCRFD face detection |
| `adaface_ir101_webface12m.onnx` | **KEEP** | ~250MB | ArcFace face embeddings |
| `minifasnet_v2.onnx` | **KEEP** | ~2MB | Liveness / anti-spoofing |
| — | `osnet_x1_0.onnx` | **ADD ~8MB** | Person RE-ID embeddings |
 
**Net change: ~120MB less VRAM, ~2ms added for RE-ID on tracked persons without faces.**
 
Update `scripts/download_models.py`:
- Remove `yolov8x-pose.onnx` download
- Add `yolov8l.onnx` download (ultralytics export, detection-only)
- Add `osnet_x1_0.onnx` download (from HuggingFace: `kaiyangzhou/osnet`)
 
---
 
## Database Schema Changes
 
### Main DB (PostgreSQL) — Alembic migration:
 
```sql
-- Track identity state improvements
ALTER TABLE sightings ADD COLUMN tracking_method VARCHAR(20) DEFAULT 'face';
-- Values: 'face', 'reid_bridge', 'face_reconfirmed'
ALTER TABLE sightings ADD COLUMN reid_bridged_cycles INTEGER DEFAULT 0;
ALTER TABLE sightings ADD COLUMN reid_bridge_start TIMESTAMPTZ;
ALTER TABLE sightings ADD COLUMN reid_bridge_end TIMESTAMPTZ;
ALTER TABLE sightings ADD COLUMN face_quality_avg FLOAT;
ALTER TABLE sightings ADD COLUMN face_quality_best FLOAT;
 
-- Best-shot gallery
CREATE TABLE best_shots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id UUID NOT NULL REFERENCES persons(id),
    session_id UUID,
    client_id UUID NOT NULL,
    quality_score FLOAT NOT NULL,
    face_crop_url TEXT NOT NULL,       -- MinIO path
    captured_at TIMESTAMPTZ NOT NULL,
    camera_id UUID REFERENCES cameras(id),
    preset_id VARCHAR(50),
    yaw_degrees FLOAT,
    pitch_degrees FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_bestshots_person_quality ON best_shots(person_id, quality_score DESC);
 
-- Activity labels (future-proof)
ALTER TABLE sightings ADD COLUMN activity_labels JSONB DEFAULT '[]';
-- Example: [{"label": "attentive", "confidence": 0.85, "at": "2025-01-01T10:00:00Z"}]
```
 
### TimescaleDB — Alembic_ts migration:
 
```sql
ALTER TABLE detection_log ADD COLUMN tracking_method VARCHAR(20) DEFAULT 'face';
ALTER TABLE detection_log ADD COLUMN reid_confidence FLOAT;
ALTER TABLE detection_log ADD COLUMN face_quality_score FLOAT;
ALTER TABLE detection_log ADD COLUMN mot_track_id INTEGER;
```
 
---
 
## Files Changed Summary
 
| File | Action | Description |
|------|--------|-------------|
| `ai_pipeline.py` | **MAJOR REWRITE** | New orchestrator: parallel detection, quality gate, identity binding, RE-ID |
| `mot_tracker.py` | **NEW** | BoT-SORT with camera motion compensation |
| `face_quality.py` | **NEW** | Landmark-based face quality assessment |
| `reid_engine.py` | **NEW** | OSNet inference + gallery management |
| `identity_state.py` | **NEW** | Track-identity binding manager |
| `best_shot.py` | **NEW** | Top-K face crop gallery per identity |
| `ptz_brain.py` | **MAJOR REWRITE** | Face-quality-aware, identity-priority targeting |
| `sighting_engine.py` | **REFACTOR** | Face-anchored sightings with RE-ID bridge continuity |
| `attendance_engine.py` | **MINOR UPDATE** | Handle new sighting metadata fields |
| `cross_camera.py` | **UPDATE** | Use OSNet RE-ID for cross-camera handoff |
| `download_models.py` | **UPDATE** | yolov8l + osnet_x1_0 instead of yolov8x-pose |
 
---
 
## Configuration Reference (All New + Changed)
 
```bash
# === Model Paths ===
YOLO_MODEL_PATH=models/yolov8l.onnx        # Changed from yolov8x-pose
YOLO_MODE=detect                             # NEW: 'detect' only, no 'pose'
REID_MODEL_PATH=models/osnet_x1_0.onnx      # NEW
 
# === Face Quality Gate ===
FACE_QUALITY_RECOGNITION_THRESH=0.55         # NEW: min quality for ArcFace
FACE_QUALITY_TRACKING_THRESH=0.25            # NEW: min quality for face tracking
FACE_QUALITY_MIN_SIZE=40                     # NEW: min face pixel size
 
# === BoT-SORT MOT Tracker ===
MOT_TRACK_HIGH_THRESH=0.5
MOT_TRACK_LOW_THRESH=0.1
MOT_MATCH_THRESH=0.8
MOT_TRACK_BUFFER=30
MOT_MIN_HITS=3
MOT_GMC_METHOD=sparseOptFlow
MOT_APPEARANCE_WEIGHT=0.2
 
# === RE-ID Engine ===
REID_MATCH_THRESHOLD=0.45
REID_ANCHOR_TTL_SECONDS=30
REID_ANCHOR_GALLERY_SIZE=5
REID_PRESET_SCOPED=true
REID_MAX_BRIDGE_SECONDS=60
 
# === Best-Shot Gallery ===
BEST_SHOT_GALLERY_SIZE=5
BEST_SHOT_MIN_QUALITY=0.6
 
# === PTZ Intelligence ===
PTZ_FACE_SEEK_ZOOM_STEP=1                   # NEW: zoom increment when seeking face
PTZ_UNRECOGNIZED_PRIORITY_WEIGHT=100         # NEW: priority score for unrecognized persons
PTZ_REID_RECONFIRM_URGENCY=2.0              # NEW: score multiplier per second of RE-ID bridge
PTZ_MIN_QUALITY_TO_MOVE_ON=0.7              # NEW: don't leave target until this quality captured
```
 
---
 
## Testing Requirements
 
| Test File | Coverage |
|-----------|----------|
| `test_mot_tracker.py` | BoT-SORT: track creation, association, GMC, track lifecycle |
| `test_face_quality.py` | Quality scoring: yaw, pitch, blur, size, composite thresholds |
| `test_reid_engine.py` | OSNet inference, gallery CRUD, matching, TTL expiry, preset scoping |
| `test_identity_state.py` | Track-identity binding, duplicate prevention, method transitions |
| `test_face_person_association.py` | Spatial association: IoU, containment, edge cases |
| `test_pipeline_integration.py` | Full frame pipeline: detection → tracking → recognition → RE-ID |
| `test_sighting_continuity.py` | face → reid_bridge → face_reconfirmed transitions |
| `test_ptz_intelligence.py` | Target selection, quality-seeking adjustments, preset transitions |
| `test_best_shot.py` | Gallery updates, quality ranking, MinIO storage |
| Update `test_attendance_engine.py` | Handle reid_bridged_cycles, face_quality fields |
 
---
 
## Future-Proofing: Activity Detection Hooks
 
The architecture is designed so activity detection can be added without another refactor:
 
**Phase 2 additions (no architecture change needed):**
 
1. **Head Pose Estimation** — Add a lightweight head pose model (e.g., WHENet, ~5MB ONNX). Run on face crops that pass quality gate. Output yaw/pitch/roll → "attentive", "distracted", "sleeping" labels. Store in `TrackIdentityState.activity_label` and `sightings.activity_labels` JSONB.
 
2. **Micro-Action Recognition** — Use person crops from the same MOT tracks to detect actions like "hand raised", "writing", "using phone". Can use a simple temporal CNN on sequences of person crops. The MOT track provides the temporal association automatically.
 
3. **Gaze Direction** — Combined with head pose + eye landmarks (from higher-res SCRFD or a dedicated eye model), determine where a person is looking. Useful for "paying attention to speaker" in classroom scenarios.
 
4. **Group Activity** — With stable track IDs and identities, detect group formations and interactions. "Person A and B are facing each other" = "conversation."
 
**All of these slot into the existing pipeline at Step 4-6** in the orchestrator, running on crops already extracted for face quality and RE-ID. No new detection pass needed.
 
---
 
## Updated CLAUDE.md Sections
 
### AI Models table:
```
| YOLOv8l        | yolov8l.onnx                 | Person detection (bbox only)        |
| SCRFD          | buffalo_l/det_10g.onnx       | Face detection                      |
| ArcFace        | adaface_ir101_webface12m.onnx| Face embeddings                     |
| MiniFASNet v2  | minifasnet_v2.onnx           | Liveness anti-spoofing              |
| OSNet x1_0     | osnet_x1_0.onnx              | Person RE-ID appearance embeddings  |
```
 
### Backend Services table (add):
```
| mot_tracker.py     | BoT-SORT multi-object tracker with camera motion compensation |
| face_quality.py    | Face image quality assessment for recognition gating         |
| reid_engine.py     | OSNet person RE-ID + appearance gallery management           |
| identity_state.py  | Track-identity binding and lifecycle management              |
| best_shot.py       | Per-identity best face crop gallery                          |
```
 
### Data Flow:
```
1. rtsp_decoder.py pulls frames from cameras
2. ai_pipeline.py runs GPU inference (parallel):
   a. YOLOv8l person detection (bbox only)
   b. SCRFD face detection with landmarks
3. Face-to-person spatial association
4. BoT-SORT MOT tracker assigns stable track_ids (with camera motion compensation)
5. Face quality gate filters candidates for recognition
6. High-quality faces → liveness → ArcFace → FAISS → identity binding to track
7. Faceless tracks → OSNet RE-ID matching against recognized-person gallery
8. PTZ brain makes face-quality-aware targeting decisions
9. Best-shot gallery updated for recognized persons
10. Sightings (face-anchored with RE-ID continuity) → attendance_engine
11. Events published to Kafka; nodes sync embeddings via face_sync.py
```

# Hướng dẫn học nhanh – Robot Aplus Tuần 9 (dành cho Nguyễn Văn Viết)

Tài liệu này giải thích các khái niệm bằng chính code trong dự án. Mỗi mục có: ý tưởng, vị trí
trong code, và cách tự kiểm tra.

---

## 1. Gymnasium environment là gì?

Một *environment* là “thế giới” mà agent tương tác. Gymnasium quy định hai hàm:

```python
obs, info = env.reset(seed=..., options=...)
obs, reward, terminated, truncated, info = env.step(action)
```

Trong dự án: `src/robot_aplus_rl/navigation_env.py`, lớp `Navigation2DEnv(gym.Env)`.

- `reset()` chọn một kịch bản (vật cản, vị trí xuất phát, đích) rồi trả về quan sát đầu tiên.
- `step(action)` di chuyển robot 0,2 s, kiểm tra va chạm/đến đích, tính reward.

Tự kiểm tra:

```bash
.venv/Scripts/python.exe -c "from robot_aplus_rl import Navigation2DEnv; e=Navigation2DEnv(); o,i=e.reset(seed=0); print(o, i['scenario_id'])"
```

## 2. Observation space và action space

**Observation** (cái robot “thấy”) – 11 số `float32`, khai báo bằng `spaces.Box`:

| Chỉ số | Ý nghĩa | Khoảng |
|---|---|---|
| 0–7 | 8 tia khoảng cách (0°, 45°, …, 315° so với hướng robot) chia cho 3,5 m | [0, 1] |
| 8 | khoảng cách tới đích chia cho đường chéo 6√2 m | [0, 1] |
| 9, 10 | sin α, cos α với α là góc tới đích so với hướng robot | [-1, 1] |

Vì sao dùng sin/cos thay vì α? Vì α nhảy từ +π sang −π (không liên tục), còn (sin α, cos α) thì
liên tục – mạng nơ-ron dễ học hơn.

Robot **không** được biết toạ độ vật cản (thông tin “đặc quyền”), chỉ biết những gì cảm biến
thật có thể đo. Điều này quan trọng để sau này chuyển sang robot thật.

**Action** – 2 số trong [-1, 1]: `(u_v, u_ω)`, đổi sang vận tốc thật trong
`action_to_velocity()`:

```python
v = (u_v + 1) / 2 * 0.35   # chỉ đi tới, 0..0.35 m/s
w = u_w * 1.5              # quay -1.5..1.5 rad/s
```

## 3. Động học robot vi sai (differential drive)

Trong `geometry.py`, hàm `unicycle_step`:

```python
x' = x + v cos(θ) dt
y' = y + v sin(θ) dt
θ' = wrap(θ + ω dt)
```

`wrap_angle` đưa góc về [-π, π). Robot có bán kính 0,18 m nên kiểm tra va chạm phải dùng
**hình tròn**, không phải điểm: `first_contact_fraction` “phồng” vật cản thêm 0,18 m rồi kiểm
tra đoạn thẳng robot đi qua trong một bước (swept test) – vì vậy robot không thể “xuyên” qua
vật cản mỏng.

Test liên quan: `tests/test_geometry.py`.

## 4. Reward shaping

`rewards.py`:

```
r = 10·(d_t − d_{t+1}) + 20·[đến đích] − 20·[va chạm] − 0.05
```

- Phần đầu: thưởng khi **tiến gần** đích, phạt khi đi xa. Tổng phần này trong một episode
  chỉ bằng `10·(d_0 − d_T)` nên agent không thể “ăn gian” bằng cách đi tới đi lui.
- +20 khi đến đích, −20 khi đâm, −0,05 mỗi bước để không lãng phí thời gian.
- Thưởng/phạt kết thúc chỉ trả **một lần**: gọi `step()` sau khi kết thúc sẽ báo lỗi.

Test: `tests/test_reward.py`.

## 5. Episode và timestep

- **Timestep (bước)**: một lần gọi `step()` = 0,2 s mô phỏng.
- **Episode**: từ `reset()` đến khi `terminated` (đến đích hoặc va chạm) hoặc `truncated`
  (hết 250 bước). `info["outcome"]` cho biết là `success`, `collision` hay `timeout`.
- “Huấn luyện 200 000 bước” nghĩa là 200 000 lần gọi `step()`, gồm hàng nghìn episode.

## 6. PPO – học on-policy

PPO (`configs/ppo.yaml`) dùng **dữ liệu vừa thu thập bằng chính chính sách hiện tại**:

1. Chạy 8 môi trường song song, mỗi cái 256 bước → 2048 bước kinh nghiệm (*rollout*).
2. Tính advantage (GAE, λ = 0,95) – hành động tốt hơn hay tệ hơn trung bình.
3. Cập nhật mạng 10 epoch trên 2048 mẫu đó, với “clip” 0,2 để chính sách không thay đổi quá
   mạnh một lần.
4. **Vứt** dữ liệu cũ, thu thập rollout mới.

Ưu: ổn định, đơn giản. Nhược: tốn nhiều bước môi trường vì dữ liệu chỉ dùng một lần.

## 7. SAC – học off-policy và replay buffer

SAC (`configs/sac.yaml`) lưu **mọi** chuyển tiếp `(s, a, r, s', done)` vào **replay buffer**
(tối đa 300 000 mẫu) và mỗi bước lấy ngẫu nhiên 256 mẫu cũ để cập nhật:

- 5 000 bước đầu: hành động ngẫu nhiên để lấp buffer (`learning_starts`).
- Hai mạng Q (critic) ước lượng giá trị; actor học hành động tối đa hoá Q + entropy.
- `ent_coef: auto` tự điều chỉnh mức “tò mò” (entropy).

Ưu: tận dụng dữ liệu tốt hơn (mỗi mẫu dùng nhiều lần). Nhược: mỗi bước có một lần cập nhật
mạng → chậm hơn nhiều về thời gian thực trên CPU (đo được ~16 ms/lần cập nhật khi chạy một
mình trên máy này).

## 8. Training loss khác episodic return như thế nào?

- **Loss** (trong `logs/progress.csv`, ví dụ `train/value_loss`, `train/actor_loss`) đo mức
  khớp của mạng với mục tiêu *đang thay đổi*. Loss giảm **không** có nghĩa robot giỏi hơn.
- **Episodic return** (tổng reward một episode, trong `training_episodes.csv`) mới phản ánh
  hành vi. Nhưng return cũng có thể cao do tiến gần đích rồi đâm (vẫn được phần thưởng tiến độ).

## 9. Evaluation khác training ra sao?

| | Training | Evaluation |
|---|---|---|
| Kịch bản | ngẫu nhiên, seed 0..10⁶ | **cố định**, seed 2 000 000+ (chưa từng dùng để huấn luyện) |
| Hành động | ngẫu nhiên (để khám phá) | **deterministic** (hành động trung bình) |
| Mục đích | cập nhật mạng | đo hiệu quả thật |

Ngoài ra có tập *validation* (seed 1 000 000+, 30 kịch bản) dùng để theo dõi trong lúc huấn
luyện và chọn `best_model.zip`; tập *test* chỉ dùng cho kết quả cuối cùng.

## 10. Vì sao success rate quan trọng hơn reward?

Mục tiêu thật là **đến đích an toàn**. Một chính sách có thể đạt reward khá cao nhờ tiến gần
đích rồi va chạm. Vì vậy báo cáo dùng:

- Success rate SR = số lần thành công / tổng số episode,
- Collision rate CR, timeout rate,
- độ dài đường đi, thời gian đến đích, SPL xấp xỉ.

Với 150 episode, SR có sai số; dự án báo cáo khoảng tin cậy Wilson 95 % (`evaluation.py`).

## 11. Checkpoint chứa gì?

`final_model.zip` / `best_model.zip` (Stable-Baselines3) là file zip gồm:

- `policy.pth`: trọng số mạng (actor, critic),
- `policy.optimizer.pth`: trạng thái optimizer Adam,
- `data`: siêu tham số, observation/action space, số bước đã chạy (`num_timesteps`).

Replay buffer của SAC **không** nằm trong zip (phải lưu riêng nếu muốn tiếp tục huấn luyện
giống hệt). Script `train.py` tự kiểm tra: nạp lại zip và so sánh hành động trên 64 quan sát
– kết quả ghi ở `run_info.json → reload_check`.

## 12. Tự tay chạy lại

```bash
cd ~/projects/robot-aplus-week9-rl
.venv/Scripts/python.exe -m pytest                                 # 45 test
.venv/Scripts/python.exe scripts/train.py --algo ppo --timesteps 4096 --label smoke --seed 0
.venv/Scripts/python.exe scripts/evaluate.py --policy heuristic
```

Sau đó xem `results/ppo/smoke_seed0/run_info.json` và `results/evaluation/episodes/heuristic.csv`.

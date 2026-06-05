"""
游戏血量 OCR 结果格式化解析器
Game HP OCR result formatting parser

处理各种 OCR 扰动：
Handles various OCR perturbations:
  - 分隔符误读:  307/350 → 307|350, 307 350, 307350
    Separator misread: 307/350 → 307|350, 307 350, 307350
  - 数字插入:   307/350 → 3071350, 3071350
    Digit insertion:  307/350 → 3071350, 3071350
  - 前缀噪声:   +307350
    Prefix noise:     +307350
  - 多检测框合并: 307|350 或 350|350 (两条结果拼在一起)
    Multi-bbox merge: 307|350 or 350|350 (two results concatenated)

核心约束:
Core constraints:
  1. current ≤ max
  2. max 极少变化（升级/换装才变），锁定后需连续 N 帧确认才更新
     max rarely changes (only on level-up/gear swap); once locked, needs N consecutive frames to confirm update
  3. current 连续变化，不会跳变太大
     current changes continuously and never jumps too much
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional


@dataclass
class HPResult:
    current: Optional[int] = None
    max_hp: Optional[int] = None
    raw: str = ""           # 原始 OCR 文本 // Raw OCR text
    confidence: str = ""    # "high" / "medium" / "low"

    @property
    def valid(self) -> bool:
        return self.current is not None and self.max_hp is not None

    def __repr__(self):
        if self.valid:
            return f"HP({self.current}/{self.max_hp}) [{self.confidence}] raw={self.raw!r}"
        return f"HP(invalid) raw={self.raw!r}"


class HPParser:
    """
    游戏血量解析器，带跨帧状态跟踪。
    Game HP parser with cross-frame state tracking.

    max_hp 锁定机制：
    max_hp locking mechanism:
      - 首次确认后锁定
        Locked after first confirmation
      - 新值需连续 confirm_frames 帧一致才替换锁定值
        New value must be consistent for confirm_frames consecutive frames to replace the locked value
      - 期间任何不一致都重置计数，旧锁定值继续生效
        Any inconsistency resets the count; old locked value remains in effect

    用法 / Usage:
        parser = HPParser()
        result = parser.parse("307|350")   # → HP(307/350)
        result = parser.parse("3071350")   # → HP(307/350) (max 锁定 / max locked)
    """

    def __init__(self, confirm_frames: int = 10, max_jump_ratio: float = 0.3,
                 min_max_hp: int = 100):
        """
        Args:
            confirm_frames: max_hp 新值需连续多少帧一致才确认变更
                Number of consecutive frames a new max_hp must be consistent before confirmed
            max_jump_ratio: current 允许的最大单帧跳变比例 (相对于 max_hp)
                Maximum allowed single-frame jump ratio for current (relative to max_hp)
            min_max_hp: max_hp 有效下限, 低于此值视为误读 (守望先锋没有低于三位数血量的英雄)
                Minimum valid max_hp, below this is treated as misread (Overwatch has no hero with <100 HP)
        """
        # 锁定的 max（置信度高、已被确认的值）
        # Locked max (high-confidence, confirmed value)
        self.locked_max: Optional[int] = None
        # 新 max 候选及连续确认计数
        # New max candidate and consecutive confirmation count
        self.max_candidate: Optional[int] = None
        self.max_candidate_count: int = 0
        self.confirm_frames = confirm_frames

        # current 历史（最近几帧，用于平滑）
        # Current history (recent frames, used for smoothing)
        self.prev_current: Optional[int] = None
        self.max_jump_ratio = max_jump_ratio
        self.min_max_hp = min_max_hp

    def parse(self, ocr_texts: list[str]) -> HPResult:
        """
        解析 OCR 识别结果列表。
        Parse a list of OCR recognition results.

        Args:
            ocr_texts: RapidOCR 返回的文本列表，如 ["307/350"] 或 ["307", "|", "350"]
                Text list returned by RapidOCR, e.g. ["307/350"] or ["307", "|", "350"]
        """
        raw = "".join(ocr_texts).strip()
        if not raw:
            return self._fallback(raw)

        result = self._parse_raw(raw)

        # max_hp 低于有效下限 → 视为误读, 丢弃本帧
        # max_hp below minimum → treat as misread, discard this frame
        if result.valid and result.max_hp < self.min_max_hp:
            return self._fallback(raw)

        # 用锁定 max 修正
        # Apply locked max correction
        result = self._apply_locked_max(result)

        # current 跳变检查
        # Current jump validation
        result = self._validate_current(result)

        # 更新 max 锁定状态
        # Update max lock state
        self._update_max_lock(result)

        # 更新 current 历史
        # Update current history
        if result.valid and result.confidence != "low":
            self.prev_current = result.current

        return result

    def _fallback(self, raw: str) -> HPResult:
        """OCR 无结果时，沿用锁定值。Use locked values when OCR returns no result."""
        if self.locked_max is not None:
            return HPResult(current=self.prev_current, max_hp=self.locked_max,
                           raw=raw, confidence="low")
        return HPResult(raw=raw, confidence="low")

    def _strip_false_one(self, value: int) -> int:
        """去除四位数首位的误读 1。Strip false leading 1 from 4-digit number.

        守望先锋没有四位数血量的英雄，OCR 误在三位数前插入 1 时：
        1307 -> 307, 1350 -> 350
        Overwatch has no hero with 4-digit HP; when OCR inserts a false leading 1:
        """
        if 1000 <= value <= 1999:
            stripped = value - 1000
            # 去掉 1 后必须仍是有效正数 / Must still be a valid positive number
            if stripped > 0:
                return stripped
        return value

    def _parse_raw(self, raw: str) -> HPResult:
        """从原始字符串解析 current/max。Parse current/max from raw string."""
        import re
        cleaned = raw.lstrip("+-= \t")

        # 策略1: 按分隔符切分 (/, |, l, I, 空格)
        # Strategy 1: split by separators (/, |, l, I, space)
        parts = re.split(r'[/|lI\s]+', cleaned)
        parts = [p for p in parts if p]

        if len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                # 去除误读的前导 1 / Strip false leading 1
                a, b = self._strip_false_one(a), self._strip_false_one(b)
                if a <= b:
                    return HPResult(current=a, max_hp=b, raw=raw, confidence="high")
                return HPResult(raw=raw, confidence="low")
            except ValueError:
                pass

        if len(parts) > 2:
            nums = []
            for p in parts:
                p = p.strip()
                if p.isdigit():
                    nums.append(self._strip_false_one(int(p)))
                if len(nums) == 2:
                    break
            if len(nums) == 2 and nums[0] <= nums[1]:
                return HPResult(current=nums[0], max_hp=nums[1], raw=raw, confidence="high")

        # 策略2: 纯数字串，尝试拆分
        # Strategy 2: pure digit string, try splitting
        digits = re.sub(r'[^0-9]', '', raw)
        if len(digits) >= 2:
            result = self._split_digits(digits, raw)
            if result and result.valid:
                return result

        return HPResult(raw=raw, confidence="low")

    def _split_digits(self, digits: str, raw: str) -> Optional[HPResult]:
        """纯数字串拆分为 current/max。Split a pure digit string into current/max."""
        # 如果有锁定 max，在数字串中查找 max 后缀
        # If locked max exists, search for max suffix in digit string
        locked = self.locked_max
        if locked is not None:
            max_str = str(locked)
            idx = digits.rfind(max_str)
            if idx > 0:
                try:
                    current = self._strip_false_one(int(digits[:idx]))
                    if current <= locked:
                        return HPResult(current=current, max_hp=locked,
                                       raw=raw, confidence="medium")
                except ValueError:
                    pass

            # max 可能被误读 1 位
            # max may be misread by 1 digit
            for extra in [-1, 1]:
                adj = locked + extra * (10 ** (len(max_str) - 1))
                if adj <= 0:
                    continue
                adj_str = str(adj)
                idx = digits.rfind(adj_str)
                if idx > 0:
                    try:
                        current = self._strip_false_one(int(digits[:idx]))
                        if 0 < current <= locked * 1.5:
                            return HPResult(current=current, max_hp=locked,
                                           raw=raw, confidence="medium")
                    except ValueError:
                        pass

        # 无锁定 max: 枚举所有拆分
        # No locked max: enumerate all splits
        n = len(digits)
        candidates = []
        for split in range(1, n):
            a_str, b_str = digits[:split], digits[split:]
            if (len(a_str) > 1 and a_str[0] == '0') or \
               (len(b_str) > 1 and b_str[0] == '0'):
                continue
            try:
                a, b = int(a_str), int(b_str)
            except ValueError:
                continue
            # 去除误读的前导 1 / Strip false leading 1
            a, b = self._strip_false_one(a), self._strip_false_one(b)
            if a <= b and a > 0 and b > 0:
                len_diff = abs(len(a_str) - len(b_str))
                candidates.append((-len_diff, a, b))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, current, max_hp = candidates[0]
            return HPResult(current=current, max_hp=max_hp, raw=raw, confidence="medium")

        return None

    def _apply_locked_max(self, result: HPResult) -> HPResult:
        """用锁定 max 强制修正解析结果。Force-correct parsed result using locked max."""
        if not result.valid:
            return result

        if self.locked_max is not None and result.max_hp != self.locked_max:
            # max 与锁定值不同 → 用锁定值覆盖，保留 current
            # max differs from locked value → override with locked value, keep current
            # 但先检查 current 是否在锁定 max 范围内
            # But first check if current is within locked max range
            if result.current <= self.locked_max:
                result.max_hp = self.locked_max
                result.confidence = "medium"
            else:
                # current > locked_max，可能是 current 被误读
                # current > locked_max, possibly a current misread
                # 沿用上帧值
                # Reuse previous frame value
                if self.prev_current is not None and self.prev_current <= self.locked_max:
                    result.current = self.prev_current
                    result.max_hp = self.locked_max
                    result.confidence = "low"
                else:
                    result.confidence = "low"

        return result

    def _validate_current(self, result: HPResult) -> HPResult:
        """current 跳变检查。Validate current against jump threshold."""
        if not result.valid or result.confidence == "low":
            return result

        if self.prev_current is not None and result.max_hp is not None:
            max_delta = result.max_hp * self.max_jump_ratio
            delta = abs(result.current - self.prev_current)
            if delta > max_delta and result.confidence != "high":
                # 跳变过大且非高置信 → 沿用上帧 current
                # Jump too large and not high confidence → reuse previous current
                result.current = self.prev_current
                result.confidence = "low"

        return result

    def _update_max_lock(self, result: HPResult):
        """更新 max 锁定状态。Update max lock state."""
        if not result.valid:
            # 无效结果重置候选计数
            # Invalid result resets candidate count
            self.max_candidate_count = 0
            self.max_candidate = None
            return

        # 锁定值未设置 → 直接接受
        # Locked value not set → accept directly
        if self.locked_max is None:
            if result.confidence != "low":
                self.locked_max = result.max_hp
            return

        # 解析出的 max 与锁定值相同 → 确认，重置候选
        # Parsed max matches locked value → confirmed, reset candidate
        if result.max_hp == self.locked_max:
            self.max_candidate = None
            self.max_candidate_count = 0
            return

        # max 与锁定值不同
        # max differs from locked value
        # 只在高置信度时考虑变更（分隔符明确的读数）
        # Only consider change at high confidence (readings with clear separators)
        if result.confidence != "high":
            self.max_candidate = None
            self.max_candidate_count = 0
            return

        # 高置信度但 max 不同 → 记为候选
        # High confidence but max differs → record as candidate
        if result.max_hp == self.max_candidate:
            self.max_candidate_count += 1
        else:
            # 新候选，重置计数
            # New candidate, reset count
            self.max_candidate = result.max_hp
            self.max_candidate_count = 1

        # 连续 N 帧一致 → 确认变更
        # N consecutive consistent frames → confirm change
        if self.max_candidate_count >= self.confirm_frames:
            self.locked_max = self.max_candidate
            self.max_candidate = None
            self.max_candidate_count = 0


# ── 测试 / Tests ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 锁定 max 抗扰动测试 ===\n")

    # 模拟: max=250 已锁定，中间混入 1250/350 等误读
    # Simulate: max=250 already locked, with misreads like 1250/350 mixed in
    parser = HPParser(confirm_frames=5)
    frames = [
        ("250/250", 250, 250),     # 首次，锁定 max=250
        ("200/250", 200, 250),
        ("2001250", 200, 250),     # max 误读为 1250 → _strip_false_one 还原为 250
        ("200|250", 200, 250),
        ("200 250", 200, 250),
        ("200/1250", 200, 250),    # max 误读 1250 → 还原为 250
        ("150/350", 150, 250),     # max=350 medium → 候选不计 (需 high 才计)
        ("150/350", 150, 250),
        ("150/250", 150, 250),
        ("150/350", 150, 250),
        ("150/350", 150, 250),
        ("150/350", 150, 250),
        ("150/350", 150, 250),
        ("150/350", 150, 250),
        ("120/250", 120, 250),     # 锁定仍是 250 (350 从未被 high 确认)
        ("120/1250", 120, 250),    # 误读还原
    ]
    for text, exp_cur, exp_max in frames:
        result = parser.parse([text])
        ok = "OK" if (result.current == exp_cur and result.max_hp == exp_max) else "FAIL"
        print(f"  {ok} input={text!r:15s} → {result}  (期望 {exp_cur}/{exp_max})")

    # 模拟连续误读：max=350，连续 20 帧都把 350 读成各种乱七八糟的
    # Simulate consecutive misreads: max=350, 20 frames of garbled 350 readings
    print("\n=== 连续误读压力测试 ===\n")
    parser2 = HPParser(confirm_frames=10)
    bad_reads = [
        "350/350",    # 锁定
        "307/350",
        "3071350",    # 乱
        "307|350",
        "307350",     # 无分隔
        "307 350",
        "312|1350",   # max 误读
        "312/350",
        "3121350",    # 又乱
        "307/350",
        "307|350",
        "+307350",
        "307l350",
        "307I350",
        "307/350",
        "287/350",
        "287|1350",   # 乱
        "287/350",
        "287350",
        "287 350",
    ]
    for text in bad_reads:
        result = parser2.parse([text])
        print(f"  input={text!r:15s} → {result}")

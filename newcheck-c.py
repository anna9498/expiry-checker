"""
newcheck-c.py
掃描「目前資料夾」內的圖片,辨識 製造日期 / 保存期限 / 有效期限,並判斷是否過期。

需要套件:
    pip install rapidocr_onnxruntime "opencv-python<5" numpy Pillow
"""
import os
import re
import calendar
from collections import Counter
from datetime import date, timedelta

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

VALID_EXT = ('.jpg', '.jpeg', '.jfif', '.png')

# ==================== 可調整的設定 ====================
# 包裝上找不到「保存期限」時使用的預設值,格式 (數字, 單位),單位: 'day' / 'month' / 'year'
# 例如 (12, 'month')。None = 不猜。
DEFAULT_SHELF = None

# 有保存期限、但沒有製造日期時,是否詢問你手動輸入製造日期
ASK_IF_MISSING = True

# 只讀到「一個沒有標示字樣的日期」時怎麼處理:
#   'ask'  = 一律詢問你手動選擇
#   'auto' = 沒讀到保存期限 -> 視為有效日期;有讀到保存期限 -> 詢問你
#   'exp'  = 一律當有效日期
#   'mfg'  = 一律當製造日期
UNLABELED_SINGLE = 'ask'

# 年份只讀到個位數時(例如「9年04月05日」)的推測基準:2010 代表 9 -> 2019
PARTIAL_YEAR_BASE = 2010

# True = 每張圖都印出 OCR 讀到的所有文字(除錯用)
SHOW_TEXT = False

# 影像處理都失敗時,依序嘗試的旋轉角度(正值 = 逆時針)
ANGLES = [-20, -15, -25, -10, 10, 20, 90, -90, 180,
          -30, 30, -45, 45, -60, 60, -75, 75, -120, 120, -135, 135, -150, 150]
# =====================================================

# ---------------- 關鍵字 ----------------
# 用 regex 寫,容許 OCR 常見的誤讀(例如「有效」被讀成「有放」「有敍」)
MFG_PAT = r'[製制]造日期|[製制]造日|造日期|生[產产]日期|生[產产]日|MFG|MFD|PRODUCTIONDATE|PRODDATE'
EXP_PAT = (r'有.日期|有.期限|有.期至|有效期?|效日期|效期|到期日?|'
           r'[賞赏]味期限|[賞赏]味期?|味期限?|保存期限|保質期限|保质期限|期限|日期至?|'
           r'EXP|BB|BEST|USEBY')
KW_RE = re.compile(f'(?P<mfg>{MFG_PAT})|(?P<exp>{EXP_PAT})')

# 保存期限:關鍵字後面接「數字(含中文數字) + 單位」
NUMERALS = '零〇一二兩两三四五六七八九十百'
SHELF_RE = re.compile(
    r'(?:保存期限|保存期間|保存期|保質期|保质期|有效期限|有效期|效期|SHELFLIFE|BESTBEFORE|BESTWITHIN|USEWITHIN)'
    r'[^\d' + NUMERALS + r']{0,8}?'
    r'(\d{1,3}|[' + NUMERALS + r']{1,5})'
    r'(個月|个月|月|年|天|日|MONTHS?|YEARS?|DAYS?)'
)

MONTHS = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
          'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

CN_DIGIT = {'零': 0, '〇': 0, '一': 1, '二': 2, '兩': 2, '两': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}

# OCR 常見的字元混淆(只在緊鄰數字時才替換)
CONFUSE = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', '|': '1'}
CONFUSE_RE = re.compile(r'(?<=\d)[OQIL|]|[OQIL|](?=\d)')

# ---------------- 日期格式 ----------------
P_YMD = re.compile(r'(20\d{2})[./\-年,](\d{1,2})[./\-月,](\d{1,2})日?')          # 2026.11.11 / 2017年01月25日
P_8 = re.compile(r'(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)')                       # 20260205
P_DMY = re.compile(r'(\d{1,2})[./\-](\d{1,2})[./\-](20\d{2})(?!\d)')            # 31/12/2019
P_ENG = re.compile(r'(\d{1,2})(' + '|'.join(MONTHS) + r')[A-Z]*[./\-]?(20\d{2})(?!\d)')  # 15JAN2024
P_PART = re.compile(r'(?<!\d)(\d{1,2})年(\d{1,2})月(\d{1,2})日')                 # 9年04月05日(年份不完整)
P_SPACED = re.compile(r'(?<!\d)(20\d{2}) (\d{1,2}) (\d{1,2})(?!\d)')            # 2025 09 22(用空格分隔)

GUESSED = set()  # 記錄「年份是推測的」日期


def cn2int(s):
    """'12' -> 12, '三十' -> 30, '十二' -> 12, '一百八十' -> 180"""
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch in CN_DIGIT:
            cur = CN_DIGIT[ch]
        elif ch == '十':
            total += (cur or 1) * 10
            cur = 0
        elif ch == '百':
            total += (cur or 1) * 100
            cur = 0
    return total + cur


def normalize_digits(s):
    for _ in range(2):
        s = CONFUSE_RE.sub(lambda m: CONFUSE[m.group()], s)
    return s


# ---------------- 日期解析 ----------------
def find_dates(raw):
    """
    從一行 OCR 文字找出所有日期,回傳 [(位置, date), ...]。
    位置是「去掉所有空白後」的字元位置,用來和關鍵字的位置比較先後。
    """
    spaced = re.sub(r'\s+', ' ', raw.strip()).upper()
    clean = spaced.replace(' ', '')
    found = {}

    def add(pos, y, m, d, guessed=False):
        try:
            dt = date(int(y), int(m), int(d))
        except ValueError:
            return
        if dt not in found:
            found[dt] = pos
            if guessed:
                GUESSED.add(dt)

    # 沒有空格干擾的版本(clean),以及修正 2Q17 -> 2017 之後的版本(norm)
    for s in (clean, normalize_digits(clean)):
        for m in P_YMD.finditer(s):
            add(m.start(), m.group(1), m.group(2), m.group(3))
        for m in P_8.finditer(s):
            add(m.start(), m.group(1), m.group(2), m.group(3))
        # 日/月/年:前面可能黏著別的數字(EC231/12/2019),所以不加 (?<!\d)
        for m in P_DMY.finditer(s):
            add(m.start(), m.group(3), m.group(2), m.group(1))

    for m in P_ENG.finditer(clean):
        add(m.start(), m.group(3), MONTHS[m.group(2)], m.group(1))

    # 年份不完整:9年04月05日
    for m in P_PART.finditer(normalize_digits(clean)):
        y = m.group(1)
        year = 2000 + int(y) if len(y) == 2 else PARTIAL_YEAR_BASE + int(y)
        add(m.start(), year, m.group(2), m.group(3), guessed=True)

    # 用空格分隔的日期:「2025 09 22 22:55」。去掉空白後會變成 2025092222:55,
    # 日期和時間黏在一起,所以要在保留空格的版本找
    for s in (spaced, normalize_digits(spaced)):
        for m in P_SPACED.finditer(s):
            pos = len(s[:m.start()].replace(' ', ''))
            add(pos, m.group(1), m.group(2), m.group(3))

    return sorted((pos, d) for d, pos in found.items())


def find_shelf(lines):
    """找保存期限,回傳 (數字, 單位) 或 None"""
    text = '|'.join(re.sub(r'\s+', '', l).upper() for l in lines)
    m = SHELF_RE.search(text)
    if not m:
        return None
    v, u = cn2int(m.group(1)), m.group(2)
    if v <= 0:
        return None
    if '月' in u or 'MONTH' in u:
        return v, 'month'
    if '年' in u or 'YEAR' in u:
        return v, 'year'
    return v, 'day'


def extract(lines):
    """
    逐行分析 OCR 文字,把日期歸類成:
      mfg  = 有「製造/MFG」字樣的日期
      exp  = 有「有效/賞味期限/EXP」字樣的日期
      free = 沒有任何標示的日期
    關鍵字和日期被 OCR 拆成上下兩行時,也會往下一行找。
    """
    info = {'mfg': [], 'exp': [], 'free': [], 'shelf': []}
    pending = None  # 上一行只有關鍵字、日期在下一行

    for line in lines:
        clean = re.sub(r'\s+', '', line).upper()
        kws = [(m.start(), m.lastgroup) for m in KW_RE.finditer(clean)]
        dates = find_dates(line)

        for pos, d in dates:
            label = None
            for kpos, ktype in kws:      # 取日期前面最近的一個關鍵字
                if kpos < pos:
                    label = ktype
            if label is None and pending:
                label = pending[0]
            info[label or 'free'].append(d)

        trailing_kw = bool(kws) and (not dates or kws[-1][0] > dates[-1][0])
        if trailing_kw and not SHELF_RE.search(clean):
            pending = [kws[-1][1], 1]    # 關鍵字在行尾,日期可能在下一行
        elif dates:
            pending = None
        elif pending:
            pending[1] -= 1
            if pending[1] < 0:
                pending = None

    shelf = find_shelf(lines)
    if shelf:
        info['shelf'].append(shelf)
    return info


def merge(a, b):
    return {k: a[k] + b[k] for k in a}


def has_any_date(info):
    return bool(info['mfg'] or info['exp'] or info['free'])


def is_complete(info):
    """已經能得出有效期限或需要使用者手動判定單一未標示日期了嗎?"""
    res = decide(info)
    return res['expiry'] is not None or res['ambiguous'] is not None


# ---------------- 日期計算 ----------------
def add_months(d, n):
    total = d.year * 12 + d.month - 1 + n
    y, m = divmod(total, 12)
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def add_shelf(d, shelf):
    v, unit = shelf
    if unit == 'day':
        return d + timedelta(days=v)
    if unit == 'month':
        return add_months(d, v)
    return add_months(d, v * 12)


def diff_ymd(a, b):
    """a <= b,回傳相差 (年, 月, 日)"""
    y, m, d = b.year - a.year, b.month - a.month, b.day - a.day
    if d < 0:
        m -= 1
        py, pm = (b.year, b.month - 1) if b.month > 1 else (b.year - 1, 12)
        d += calendar.monthrange(py, pm)[1]
    if m < 0:
        y -= 1
        m += 12
    return y, m, d


def approx(t):
    """相差一個月以上才顯示「(約 X 年 X 個月 X 天)」"""
    y, m, d = t
    if not (y or m):
        return ''
    parts = []
    if y:
        parts.append(f'{y} 年')
    if m:
        parts.append(f'{m} 個月')
    if d:
        parts.append(f'{d} 天')
    return '(約 ' + ' '.join(parts) + ')'


UNIT_ZH = {'day': '天', 'month': '個月', 'year': '年'}


def pick(dates, late):
    """多次 OCR 結果中,取出現次數最多的;次數一樣時取較晚(late=True)或較早的"""
    if not dates:
        return None
    cnt = Counter(dates)
    return max(cnt, key=lambda d: (cnt[d], d.toordinal() if late else -d.toordinal()))


def decide(info):
    """整合所有線索,決定 製造日期 / 保存期限 / 有效期限"""
    notes = []
    ambiguous = None
    mfg = pick(info['mfg'], late=False)
    exp = pick(info['exp'], late=True)
    shelf = info['shelf'][0] if info['shelf'] else None
    if shelf is None and DEFAULT_SHELF:
        shelf = DEFAULT_SHELF
        notes.append('包裝上沒讀到保存期限,使用程式內的預設值')
    # 同一個日期若在別輪被標示過,就不算「沒標示」
    free = sorted(d for d in set(info['free']) if d != mfg and d != exp)

    if free:
        if mfg is None and exp is not None:
            earlier = [d for d in free if d < exp]
            if earlier:
                mfg = earlier[0]
                notes.append('沒讀到「製造」字樣,取比有效日期早的日期當製造日期')
        elif mfg is not None and exp is None and not shelf:
            later = [d for d in free if d > mfg]
            if later:
                exp = later[-1]
                notes.append('沒讀到「有效」字樣,取比製造日期晚的日期當有效日期')
        elif mfg is None and exp is None:
            if len(free) >= 2:
                mfg, exp = free[0], free[-1]
                notes.append('日期都沒有標示字樣,較早的視為製造日期、較晚的視為有效日期')
            else:
                mode = UNLABELED_SINGLE
                if mode == 'auto':
                    mode = 'ask' if shelf else 'exp'
                if mode == 'mfg':
                    mfg = free[0]
                    notes.append('只有一個未標示的日期,視為製造日期')
                elif mode == 'exp':
                    exp = free[0]
                    notes.append('只有一個未標示的日期,視為有效日期(推測)')
                else:
                    ambiguous = free[0]

    derived = add_shelf(mfg, shelf) if (mfg and shelf) else None
    if exp and derived and exp != derived:
        notes.append(f'包裝標示的有效日期 {exp} 與推算值 {derived} 不同,以包裝標示為準')
    expiry = exp or derived

    if (mfg in GUESSED) or (exp in GUESSED):
        notes.append('⚠️ 日期的年份沒有完整讀到,年份是推測的,請人工確認')

    return {'mfg': mfg, 'shelf': shelf, 'expiry': expiry,
            'notes': notes, 'ambiguous': ambiguous}


# ---------------- 影像處理與 OCR ----------------
_engine = None


def ocr_lines(src):
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    result, _ = _engine(src)
    return [line[1] for line in result] if result else []


def load_img(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)  # 支援中文檔名


def rotate(g, ang):
    h, w = g.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(g, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_variants(img):
    """依序產生不同的影像處理版本,每個都是 (名稱, 影像, 是否為旋轉版本)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    yield '點陣字增強', cv2.erode(enhanced, np.ones((2, 2), np.uint8), iterations=1), False

    scale = 2 if max(gray.shape) < 1500 else 1
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    yield '放大+對比增強', clahe.apply(big), False

    yield '反相', 255 - enhanced, False

    blur = cv2.GaussianBlur(big, (3, 3), 0)
    yield '二值化', cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 10), False

    # 金屬壓印/雷射刻字:取出比周圍亮或暗的細線條
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    big2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for name, op in (('頂帽(亮字)', cv2.MORPH_TOPHAT), ('黑帽(暗字)', cv2.MORPH_BLACKHAT)):
        th = cv2.morphologyEx(big2, op, k)
        yield name, 255 - cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX), False

    # 傾斜或橫躺的文字:先轉正,再把點連成筆畫
    for ang in ANGLES:
        r = clahe.apply(rotate(big, ang))
        r = cv2.erode(r, np.ones((3, 3), np.uint8), iterations=1)  # 讓黑點變大
        r = cv2.GaussianBlur(r, (3, 3), 0)                          # 讓相鄰的點連在一起
        yield f'轉正{ang}度+連點', r, True


def try_variants(info, lines, img):
    """依序嘗試各種影像處理,資訊夠了就停。旋轉版本只在『完全沒讀到任何日期』時才會跑"""
    for name, v, is_rot in enhance_variants(img):
        if is_rot and has_any_date(info):
            break
        print(f'  [資訊不足,嘗試:{name}...]')
        l2 = ocr_lines(v)
        info = merge(info, extract(l2))
        lines += [f'({name}){t}' for t in l2]
        if is_complete(info):
            break
    return info


def select_roi(img):
    """跳出視窗:拖曳框選日期區域,放開滑鼠後按 Enter 確認,Esc 取消。回傳放大 3 倍的裁切圖"""
    import tkinter as tk
    from PIL import Image, ImageTk

    h, w = img.shape[:2]
    s = min(1.0, 900 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s)
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    root = tk.Tk()
    root.title('拖曳框選日期區域,放開滑鼠後按 Enter 確認,Esc 取消')
    photo = ImageTk.PhotoImage(pil)
    canvas = tk.Canvas(root, width=pil.width, height=pil.height, cursor='cross')
    canvas.pack()
    canvas.create_image(0, 0, anchor='nw', image=photo)

    st = {'start': None, 'rect': None, 'box': None}

    def on_down(e):
        st['start'] = (e.x, e.y)
        st['box'] = None
        if st['rect']:
            canvas.delete(st['rect'])
        st['rect'] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline='red', width=2)

    def on_move(e):
        if st['start']:
            canvas.coords(st['rect'], st['start'][0], st['start'][1], e.x, e.y)

    def on_up(e):
        if st['start']:
            x1, y1 = st['start']
            st['box'] = (min(x1, e.x), min(y1, e.y), max(x1, e.x), max(y1, e.y))

    def on_enter(_):
        root.destroy()

    def on_esc(_):
        st['box'] = None
        root.destroy()

    canvas.bind('<ButtonPress-1>', on_down)
    canvas.bind('<B1-Motion>', on_move)
    canvas.bind('<ButtonRelease-1>', on_up)
    root.bind('<Return>', on_enter)
    root.bind('<Escape>', on_esc)
    root.mainloop()

    box = st['box']
    if not box or box[2] - box[0] < 5 or box[3] - box[1] < 5:
        return None
    x1, y1, x2, y2 = [int(v / s) for v in box]
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


# ---------------- 互動 ----------------
def resolve_ambiguous(info, res, lines):
    """只讀到一個沒有標示的日期時,顯示來源文字並詢問是製造日還是有效日"""
    d = res['ambiguous']
    if not d:
        return res
    print(f'  ❓ 只找到一個沒有標示的日期:{d}')
    shown = set()
    for t in lines:
        if t not in shown and d in [x for _, x in find_dates(t)]:
            shown.add(t)
            print(f'     來源文字:{t}')
    ans = input('     請選擇這是 (1)製造日期 (2)有效日期 (Enter)跳過/不是日期:').strip()
    if ans in ('1', '2'):
        info['free'] = [x for x in info['free'] if x != d]
        info['mfg' if ans == '1' else 'exp'].append(d)
        res = decide(info)
        res['notes'].append('日期類型由手動指定')
    return res


def ask_manual_mfg():
    try:
        s = input('  ✍️ 包裝上沒讀到製造日期,可手動輸入 (例 2025-03-01,直接 Enter 跳過):').strip()
    except EOFError:
        return None
    if not s:
        return None
    ds = find_dates(s)
    if not ds:
        print('  格式看不懂,已跳過')
        return None
    return ds[0][1]


# ---------------- 主流程 ----------------
def analyze(img_name, today):
    print(f'=== 正在分析:{img_name} ===')
    GUESSED.clear()

    lines = ocr_lines(img_name)
    info = extract(lines)
    if not is_complete(info):
        info = try_variants(info, lines, load_img(img_name))

    res = resolve_ambiguous(info, decide(info), lines)

    # 什麼都沒讀到 -> 讓你框選日期區域重讀
    if not (res['expiry'] or res['mfg'] or res['shelf'] or res['ambiguous']):
        if input('  要手動框選日期區域重新辨識嗎?(y/N):').strip().lower() == 'y':
            crop = select_roi(load_img(img_name))
            if crop is not None:
                lines3 = ocr_lines(crop)
                lines += [f'(框選){t}' for t in lines3]
                info = merge(info, extract(lines3))
                if not is_complete(info):
                    info = try_variants(info, lines, crop)
                res = resolve_ambiguous(info, decide(info), lines)

    # 有保存期限、沒製造日期 -> 讓你手動補
    if ASK_IF_MISSING and res['shelf'] and not res['mfg'] and not res['expiry']:
        m = ask_manual_mfg()
        if m:
            info['mfg'].append(m)
            res = decide(info)
            res['notes'].append('製造日期為手動輸入')

    if res['mfg']:
        print(f"  📅 製造日期:{res['mfg']}")
    if res['shelf']:
        v, u = res['shelf']
        print(f'  ⏳ 保存期限:{v} {UNIT_ZH[u]}')

    if res['expiry']:
        print(f"  📆 有效期限:{res['expiry']}")
        days = (res['expiry'] - today).days
        if days < 0:
            print(f'  ❌ 已過期 {-days} 天{approx(diff_ymd(res["expiry"], today))}')
        elif days == 0:
            print('  ⚠️ 今天到期')
        else:
            print(f'  ✅ 尚未過期,還剩 {days} 天{approx(diff_ymd(today, res["expiry"]))}')
    elif res['shelf']:
        print('  ⚠️ 有保存期限,但沒有製造日期,無法推算有效期限')
    elif res['mfg']:
        print('  ⚠️ 有製造日期,但讀不到保存期限,無法推算有效期限'
              '(可在程式最上方設定 DEFAULT_SHELF)')
    else:
        print('  ❌ 無法判定日期')

    for n in res['notes']:
        print(f'  ℹ️ {n}')

    if SHOW_TEXT or not res['expiry']:
        print(f'  🔍 AI 讀到的所有文字:{lines}')
    print('-' * 45)


def main():
    image_files = sorted(f for f in os.listdir('.') if f.lower().endswith(VALID_EXT))
    today = date.today()
    for img_name in image_files:
        analyze(img_name, today)


if __name__ == '__main__':
    main()
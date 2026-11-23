import os
import re
import calendar
from collections import Counter
from datetime import date, timedelta

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
VALID_EXT = ('.jpg', '.jpeg', '.jfif', '.png')

# 包裝上找不到「保存期限」時使用的預設值,格式 (數字, 單位),單位: 'day' / 'month' / 'year'
# 例如 (12, 'month')。設成 None 代表不猜。
DEFAULT_SHELF = None

# 有保存期限、但包裝沒有製造日期時,是否詢問你手動輸入製造日期
ASK_IF_MISSING = True

# 年份只讀到個位數時(例如「9年04月05日」)的推測基準:2010 代表 9 -> 2019
PARTIAL_YEAR_BASE = 2010

# 只讀到「一個沒有標示字樣的日期」時怎麼處理:
# 'ask' = 詢問你 / 'mfg' = 當製造日期 / 'exp' = 當有效日期
UNLABELED_SINGLE = 'ask'

# ---------------- 關鍵字 ----------------
MFG_KW = ['製造日期', '製造日', '造日期', '生產日期', '生產日', '生产日期', '生产日',
          'MFG', 'MFD', 'PRODUCTIONDATE', 'PRODDATE']
EXP_KW = ['有效日期', '有效期限', '有效期至', '有效期', '有效', '效期', '到期日', '到期',
          '效日期', '文日期', '有收日期', '有收', '日期至',
          '賞味期限', '赏味期限', 'EXP', 'BB', 'BEST', 'USEBY']
KW_RE = re.compile('|'.join(sorted(map(re.escape, MFG_KW + EXP_KW), key=len, reverse=True)))

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
def find_dates(clean):
    """從已去空白、轉大寫的文字中找出所有日期,回傳 [(位置, date), ...]"""
    norm = normalize_digits(clean)  # 修正 2Q17 -> 2017 這類誤讀
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

    for s in (clean, norm):
        # 年-月-日:2026.11.11 / 2017年01月25日
        for m in re.finditer(r'(20\d{2})[./\-年](\d{1,2})[./\-月](\d{1,2})日?', s):
            add(m.start(), m.group(1), m.group(2), m.group(3))
        # 純 8 碼:20260205
        for m in re.finditer(r'(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)', s):
            add(m.start(), m.group(1), m.group(2), m.group(3))
        # 日/月/年:31/12/2019(前面可能黏著別的數字,例如 EC231/12/2019)
        for m in re.finditer(r'(\d{1,2})[./\-](\d{1,2})[./\-](20\d{2})(?!\d)', s):
            add(m.start(), m.group(3), m.group(2), m.group(1))

    # 英文月份:15JAN2024
    pat = r'(\d{1,2})(' + '|'.join(MONTHS) + r')[A-Z]*[./\-]?(20\d{2})(?!\d)'
    for m in re.finditer(pat, clean):
        add(m.start(), m.group(3), MONTHS[m.group(2)], m.group(1))

    # 年份不完整:19年04月05日 / 9年04月05日(OCR 把年份前幾位漏掉)
    for m in re.finditer(r'(?<!\d)(\d{1,2})年(\d{1,2})月(\d{1,2})日', norm):
        y = m.group(1)
        year = 2000 + int(y) if len(y) == 2 else PARTIAL_YEAR_BASE + int(y)
        add(m.start(), year, m.group(2), m.group(3), guessed=True)

    return sorted((pos, d) for d, pos in found.items())


def find_shelf(lines):
    """找保存期限,回傳 (數字, 單位) 或 None"""
    text = '|'.join(l.replace(' ', '').upper() for l in lines)
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
      exp  = 有「有效/EXP」字樣的日期
      free = 沒有任何標示的日期
    關鍵字和日期被 OCR 拆成上下兩行時,也會往下一行找。
    """
    info = {'mfg': [], 'exp': [], 'free': [], 'shelf': []}
    pending = None

    for line in lines:
        clean = line.replace(' ', '').upper()
        kws = [(m.start(), 'mfg' if m.group() in MFG_KW else 'exp')
               for m in KW_RE.finditer(clean)]
        dates = find_dates(clean)

        for pos, d in dates:
            label = None
            for kpos, ktype in kws:
                if kpos < pos:
                    label = ktype
            if label is None and pending:
                label = pending[0]
            info[label or 'free'].append(d)

        if kws and (not dates or kws[-1][0] > dates[-1][0]):
            pending = [kws[-1][1], 1]
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


def is_complete(info):
    """資訊夠不夠推出有效期限?不夠就換下一種影像處理再讀一次"""
    if info['exp']:
        return True
    return bool(info['mfg'] and (info['shelf'] or DEFAULT_SHELF))


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
    """相差一個月以上才顯示「約 X 年 X 個月 X 天」"""
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
    free = sorted(set(info['free']))

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
            elif UNLABELED_SINGLE == 'mfg':
                mfg = free[0]
                notes.append('只有一個未標示的日期,依設定視為製造日期')
            elif UNLABELED_SINGLE == 'exp':
                exp = free[0]
                notes.append('只有一個未標示的日期,依設定視為有效日期')
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


def ask_manual_mfg():
    try:
        s = input('  ✍️ 包裝上沒讀到製造日期,可手動輸入 (例 2025-03-01,直接 Enter 跳過):').strip()
    except EOFError:
        return None
    if not s:
        return None
    ds = find_dates(s.replace(' ', '').upper())
    if not ds:
        print('  格式看不懂,已跳過')
        return None
    return ds[0][1]


# ---------------- 影像處理與 OCR ----------------
def load_img(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)  # 支援中文檔名


def enhance_variants(img):
    """依序產生不同的影像處理版本,每個都是 (名稱, 影像)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    yield '點陣字增強', cv2.erode(enhanced, np.ones((2, 2), np.uint8), iterations=1)

    scale = 2 if max(gray.shape) < 1500 else 1
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    yield '放大+對比增強', clahe.apply(big)

    yield '反相', 255 - enhanced

    blur = cv2.GaussianBlur(big, (3, 3), 0)
    yield '二值化', cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 10)

        # 5. 金屬壓印/雷射刻字:取出比周圍亮或暗的細線條
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    big2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for name, op in (('頂帽(亮字)', cv2.MORPH_TOPHAT), ('黑帽(暗字)', cv2.MORPH_BLACKHAT)):
        th = cv2.morphologyEx(big2, op, k)
        yield name, 255 - cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
        
    # 6. 傾斜的點陣字:先把圖轉正,再把點連成筆畫
    def rotate(g, ang):
        h, w = g.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        return cv2.warpAffine(g, M, (nw, nh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    for ang in (-20, -15, -25, -10, 10, 20):
        r = clahe.apply(rotate(big, ang))
        r = cv2.erode(r, np.ones((3, 3), np.uint8), iterations=1)  # 讓黑點變大
        r = cv2.GaussianBlur(r, (3, 3), 0)                          # 讓相鄰的點連在一起
        yield f'轉正{ang}度+連點', r


def try_variants(info, lines, img):
    """依序嘗試各種影像處理,資訊夠了就停"""
    for name, v in enhance_variants(img):
        print(f'  [資訊不足,嘗試:{name}...]')
        l2 = ocr_lines(v)
        info = merge(info, extract(l2))
        lines += [f'({name}){t}' for t in l2]
        if is_complete(info):
            break
    return info


def select_roi(img):
    """跳出視窗讓你用滑鼠框選日期區域,回傳放大 3 倍的裁切圖"""
    h, w = img.shape[:2]
    s = min(1.0, 1000 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s)
    x, y, rw, rh = cv2.selectROI('Drag to select date area, Enter=OK, c=cancel',
                                 small, showCrosshair=False)
    cv2.destroyAllWindows()
    if rw == 0 or rh == 0:
        return None
    crop = img[int(y / s):int((y + rh) / s), int(x / s):int((x + rw) / s)]
    return cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def resolve_ambiguous(info, res, lines):
    """只讀到一個沒標示的日期時,顯示來源文字並詢問是製造日還是有效日"""
    d = res['ambiguous']
    if not d:
        return res
    print(f'  ❓ 讀到一個沒有標示的日期:{d}')
    for t in lines:
        if d in [x for _, x in find_dates(t.replace(' ', '').upper())]:
            print(f'     來源文字:{t}')
    ans = input('     這是 (1)製造日期 (2)有效日期 (Enter)不是日期/跳過:').strip()
    if ans in ('1', '2'):
        info['free'] = [x for x in info['free'] if x != d]
        info['mfg' if ans == '1' else 'exp'].append(d)
        res = decide(info)
        res['notes'].append('日期類型由手動指定')
    return res


def ocr_lines(src):
    result, _ = engine(src)
    return [line[1] for line in result] if result else []

for t in ['EC231/12/2019', '2021.09.10', '20260205', '2017年01月25日', '9年04月05日']:
    print(t, '->', [str(d) for _, d in find_dates(t)])
# ---------------- 主流程 ----------------
image_files = [f for f in os.listdir('.') if f.lower().endswith(VALID_EXT)]
today = date.today()

for img_name in image_files:
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

    if not res['expiry']:
        print(f'  🔍 AI 讀到的所有文字:{lines}')
    print('-' * 45)
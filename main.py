import os
import json
import re
import time
import requests
import opencc
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# --- 全局配置与初始化 ---
timestart = datetime.now()
print(f"🚀 IPTV频道分类系统启动 @ {timestart.strftime('%Y-%m-%d %H:%M:%S')}")

# --- 修复 OpenCC 加载问题 ---
try:
    CC_CONVERTER = opencc.OpenCC('t2s')
    print("✅ OpenCC转换器已成功加载内置配置 't2s'")
except Exception as e:
    print(f"❌ 错误: 加载OpenCC转换器失败 - {e}")
    class FallbackConverter:
        def convert(self, text):
            trad_to_simp = {
                '廣': '广', '東': '东', '衛': '卫', '視': '视', '臺': '台',
                '灣': '湾', '電': '电', '視': '视', '頻': '频', '綜': '综',
                '藝': '艺', '體': '体', '育': '育', '國': '国', '際': '际'
            }
            return ''.join(trad_to_simp.get(char, char) for char in text)
    CC_CONVERTER = FallbackConverter()
    print("✅ 使用简易简繁转换器")

# 加载分类配置
with open('category_config.json', 'r', encoding='utf-8') as f:
    CATEGORY_CONFIG = json.load(f)

# 配置 requests 会话
def create_requests_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    retries = Retry(total=5, backoff_factor=0.5, 
                   status_forcelist=[500, 502, 503, 504, 429],
                   allowed_methods=frozenset(['GET', 'POST']))
    adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

HTTP_SESSION = create_requests_session()

# 频道名称规范化配置
REMOVAL_LIST = [
    "「IPV4」", "「IPV6」", "[ipv6]", "[ipv4]", "_电信", "电信", "（HD）", "[超清]", 
    "高清", "超清", "-HD", "(HK)", "AKtv", "@", "IPV6", "🎞️", "🎦", " ", "[BD]", 
    "[VGA]", "[HD]", "[SD]", "(1080p)", "(720p)", "(480p)", "HD", "FHD", "4K", 
    "8K", "直播", "LIVE", "实时", "RTMP", "HLS", "HTTP", "://", "转码", "流畅", "测试"
]
REPLACEMENTS = {
    "CCTV-": "CCTV", "CCTV0": "CCTV", "PLUS": "+", "NewTV-": "NewTV", 
    "iHOT-": "iHOT", "NEW": "New", "New_": "New", "中央": "CCTV", "广东体育卫视": "广东体育",
    "湖南电视台": "湖南卫视", "浙江电视台": "浙江卫视", "東方衛視": "东方卫视", "凤凰中文": "凤凰卫视",
    "凤凰资讯台": "凤凰资讯", "FOX体育": "FS", "ESPN体育": "ESPN", "DISCOVERY探索": "探索频道",
    "国家地理频道": "国家地理", "历史频道": "历史", "HBO电影": "HBO", "CNN国际": "CNN"
}

# 频道名称规范化正则
CHANNEL_PATTERNS = [
    (r'^(CCTV[\d+]+)(?:[\u4e00-\u9fa5]*)$', r'\1'),  # CCTV数字频道
    (r'(湖南|浙江|江苏|东方|北京|广东|深圳|天津|山东)(?:卫视|电视台)', r'\1卫视'),
    (r'(凤凰)(?:中文|资讯|卫视)', r'\1卫视'),
    (r'(星空|华娱|TVB|翡翠|明珠)(?:卫视|台)', r'\1卫视'),
    (r'(HBO|CNN|BBC|NHK|DISCOVERY)(?:\s*[\u4e00-\u9fa5]+)?$', r'\1'),
    (r'(卫视|体育|新闻|电影|娱乐|卡通|国际|综合)(?:频道|台)', r'\1频道'),
]

# --- 核心功能函数 ---

def read_txt_to_array(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"读取文件 '{file_name}' 出错: {e}")
        return []

def load_corrections(filename):
    corrections = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        correct_name = parts[0].strip()
                        for name in parts[1:]:
                            corrections[name.strip()] = correct_name
    except Exception as e:
        print(f"加载纠错文件 '{filename}' 出错: {e}")
    return corrections

def fetch_source_content(url):
    try:
        response = HTTP_SESSION.get(url, timeout=15)
        response.raise_for_status()
        
        content = response.content
        text = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'iso-8859-1', 'big5']:
            try:
                text = content.decode(encoding)
                if text.strip().startswith("#EXTM3U"):
                    text = convert_m3u_to_txt(text)
                return text
            except UnicodeDecodeError:
                continue
        print(f"警告: 无法解码源 {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"处理URL源出错: {url} ({e})")
        return None

def convert_m3u_to_txt(m3u_content):
    lines = m3u_content.strip().split('\n')
    txt_lines = []
    channel_name = ""
    for line in lines:
        line = line.strip()
        if line.startswith("#EXTM3U"):
            continue
        if line.startswith("#EXTINF"):
            # 提取频道名称，去除可能的参数
            name_part = line.split(',', 1)[-1]
            channel_name = re.sub(r'[^,\w\s\u4e00-\u9fa5]+', '', name_part).strip()
        elif line.startswith("http") or line.startswith("rtmp"):
            if channel_name:
                txt_lines.append(f"{channel_name},{line}")
                channel_name = ""
        elif "#genre#" not in line and "," in line and re.match(r'^[^,]+,[^\s]+://[^\s]+$', line):
            txt_lines.append(line)
    return '\n'.join(txt_lines)

def normalize_channel_name(channel_name):
    """深度规范化频道名称"""
    # 简繁转换
    try:
        simplified_name = CC_CONVERTER.convert(channel_name)
    except Exception:
        simplified_name = channel_name
    
    # 移除无用字符和词语
    cleaned_name = simplified_name
    for item in REMOVAL_LIST:
        cleaned_name = cleaned_name.replace(item, "")
    
    # 应用正则规范化
    for pattern, replacement in CHANNEL_PATTERNS:
        cleaned_name = re.sub(pattern, replacement, cleaned_name)
    
    # 应用替换规则
    for old, new in REPLACEMENTS.items():
        cleaned_name = cleaned_name.replace(old, new)
    
    # 统一格式
    cleaned_name = re.sub(r'\s+', '', cleaned_name)  # 移除空格
    cleaned_name = re.sub(r'[^\w\u4e00-\u9fa5+]', '', cleaned_name)  # 移除非中文/英文/数字字符
    
    return cleaned_name.strip()

def parse_and_clean_channels(source_content, corrections_name):
    channels = {}
    if not source_content:
        return channels
    
    lines = source_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or "#genre#" in line or "#EXTINF:" in line:
            continue
        
        try:
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
                
            raw_name = parts[0].strip()
            channel_url = parts[1].split('$')[0].strip()

            if not channel_url:
                continue
            
            # 深度规范化名称
            cleaned_name = normalize_channel_name(raw_name)
            
            # 应用纠错词典
            corrected_name = corrections_name.get(cleaned_name, cleaned_name)
            
            # 合并相同频道不同来源
            if corrected_name in channels:
                # 保留更长的URL（通常更完整）
                if len(channel_url) > len(channels[corrected_name]):
                    channels[corrected_name] = channel_url
            else:
                channels[corrected_name] = channel_url
                
        except Exception as e:
            print(f"解析频道行失败: {line} - {e}")
    return channels

def check_channel_availability(channel_info, timeout=2):
    """检查频道可用性并返回延迟"""
    name, url = channel_info
    if "127.0.0.1" in url or "localhost" in url:
        return name, url, 0

    try:
        # 对于可能的大文件流，使用HEAD方法可能不合适，改用带范围的GET
        headers = {'Range': 'bytes=0-1', 'Connection': 'close'} if any(ext in url for ext in ['.m3u8', '.flv', '.ts']) else {}
        
        start_time = time.time()
        response = HTTP_SESSION.get(url, headers=headers, timeout=timeout, stream=True, verify=False)
        response.close()  # 立即关闭连接
        end_time = time.time()
        
        latency = int((end_time - start_time) * 1000)
        if response.status_code < 400 or response.status_code == 416:  # 416表示范围请求成功
            return name, url, latency
        return name, url, -1
    except Exception:
        return name, url, -1

def classify_channel(channel_name):
    """智能分类频道"""
    channel_name_lower = channel_name.lower()
    
    # 特殊频道优先处理
    if "春晚" in channel_name: return "cw"
    if "直播中国" in channel_name: return "zb"
    if any(kw in channel_name_lower for kw in ["mtv", "music", "音樂", "演唱会"]): return "mv"
    if any(kw in channel_name_lower for kw in ["radio", "广播", "fm", "am"]): return "radio"
    if any(kw in channel_name for kw in ["回看", "重播", "回放", "录像"]): return "lx"
    
    # 遍历所有分类配置
    for category_type in CATEGORY_CONFIG.values():
        for category_id, info in category_type.items():
            # 先检查字典中的完整频道名称
            for dict_name in info.get("dictionary", []):
                if dict_name in channel_name:
                    return category_id
            
            # 再检查关键词
            for keyword in info.get("keywords", []):
                if keyword in channel_name_lower:
                    return category_id
                    
    return "other"

def sort_data(order, data):
    order_dict = {name: i for i, name in enumerate(order)}
    return sorted(data, key=lambda line: order_dict.get(line.split(',')[0], len(order)))

def save_files(categorized_lists):
    all_lines = []
    utc_time = datetime.now(timezone.utc)
    beijing_time = utc_time + timedelta(hours=8)
    version = f"{beijing_time.strftime('%Y%m%d %H:%M')},https://gcalic.v.myalicdn.com/gc/wgw05_1/index.m3u8?contentid=2820180516001"
    all_lines.extend([f"更新时间,#genre#", version, ''])

    total_channels = 0

    def add_category(name, lines, dictionary=None):
        nonlocal total_channels
        if lines:
            all_lines.append(f"{name},#genre#")
            sorted_lines = sort_data(dictionary, lines) if dictionary else sorted(lines)
            all_lines.extend(sorted_lines)
            all_lines.append('')
            total_channels += len(lines)

    # 按分类优先级输出
    for category_type in CATEGORY_CONFIG.values():
        for cat_id, info in category_type.items():
            if cat_id in categorized_lists and categorized_lists[cat_id]:
                add_category(info["name"], categorized_lists[cat_id], info.get("dictionary"))
    
    # 特殊类别
    add_category("春晚", categorized_lists.get('cw', []))
    add_category("直播中国", categorized_lists.get('zb', []))
    add_category("音乐MV", categorized_lists.get('mv', []))
    add_category("收音机频道", categorized_lists.get('radio', []))
    add_category("录像回放", categorized_lists.get('lx', []))
    add_category("其他频道", categorized_lists.get('other', []))

    final_content = "\n".join(all_lines)

    try:
        with open("live.txt", "w", encoding='utf-8') as f:
            f.write(final_content)
        print("✅ 频道文件已保存: live.txt")
    except Exception as e:
        print(f"❌ 保存文件出错: {e}")

    # 生成M3U文件
    try:
        m3u_content = '#EXTM3U x-tvg-url="https://epg.112114.xyz/pp.xml.gz"\n'
        group_title = ""
        for line in final_content.strip().split('\n'):
            if not line.strip(): continue
            if "#genre#" in line:
                group_title = line.split(',')[0]
                continue
            if "," in line:
                name, url = line.split(',', 1)
                logo = f"https://epg.112114.xyz/logo/{name}.png"
                m3u_content += f'#EXTINF:-1 tvg-name="{name}" tvg-logo="{logo}" group-title="{group_title}",{name}\n{url}\n'
        
        with open("live.m3u", "w", encoding='utf-8') as f:
            f.write(m3u_content)
        print("✅ M3U文件已保存: live.m3u")
    except Exception as e:
        print(f"❌ 生成M3U文件出错: {e}")

    return total_channels

# --- 主执行流程 ---

def main():
    print("➡️ 步骤 1/5: 加载本地资源...")
    assets_dir = 'assets'
    os.makedirs(assets_dir, exist_ok=True)
    
    urls_file = os.path.join(assets_dir, 'urls.txt')
    corrections_file = os.path.join(assets_dir, 'corrections_name.txt')

    # 如果文件不存在则创建
    if not os.path.exists(urls_file):
        with open(urls_file, 'w', encoding='utf-8') as f:
            f.write("# 默认源\nhttps://gcalic.v.myalicdn.com/gc/wgw05_1/index.m3u8\n")
    
    if not os.path.exists(corrections_file):
        with open(corrections_file, 'w', encoding='utf-8') as f:
            f.write("# 频道名称纠错文件\nCCTV1,央视1台,中央1台\n湖南卫视,湖南电视台\n")

    urls_to_process = read_txt_to_array(urls_file)
    corrections = load_corrections(corrections_file)

    if not urls_to_process:
        print(f"❌ 错误: '{urls_file}' 为空或不存在，程序退出。")
        return

    print(f"\n➡️ 步骤 2/5: 获取 {len(urls_to_process)} 个在线源...")
    all_raw_channels = {}
    for url in tqdm(urls_to_process, desc="处理源"):
        try:
            content = fetch_source_content(url)
            if content:
                parsed_channels = parse_and_clean_channels(content, corrections)
                # 合并相同频道
                for name, url in parsed_channels.items():
                    if name in all_raw_channels:
                        # 保留更长的URL
                        if len(url) > len(all_raw_channels[name]):
                            all_raw_channels[name] = url
                    else:
                        all_raw_channels[name] = url
        except Exception as e:
            print(f"❌ 处理源失败: {url} - {e}")
    
    print(f"\n✅ 成功获取并解析了 {len(all_raw_channels)} 个不重复的频道。")

    if not all_raw_channels:
        print("❌ 错误: 没有获取到任何频道，程序退出")
        return

    print(f"\n➡️ 步骤 3/5: 检测 {len(all_raw_channels)} 个频道的有效性...")
    valid_channels = []
    total = len(all_raw_channels)
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(check_channel_availability, (name, url)): (name, url) 
                   for name, url in all_raw_channels.items()}
        
        for future in tqdm(as_completed(futures), total=total, desc="检测频道"):
            try:
                name, url, latency = future.result()
                if latency >= 0:
                    valid_channels.append((name, url))
            except Exception:
                pass
    
    print(f"\n✅ 检测完成，发现 {len(valid_channels)} 个有效频道。")

    print("\n➡️ 步骤 4/5: 智能分类频道...")
    categorized_lists = {}
    
    for name, url in tqdm(valid_channels, desc="分类频道"):
        category = classify_channel(name)
        if category not in categorized_lists:
            categorized_lists[category] = []
        categorized_lists[category].append(f"{name},{url}")
    
    # 打印分类统计
    print("\n📊 分类统计:")
    for cat_id, channels in categorized_lists.items():
        cat_name = "其他"
        for category_type in CATEGORY_CONFIG.values():
            for cid, info in category_type.items():
                if cid == cat_id:
                    cat_name = info["name"]
                    break
        print(f"  - {cat_name}: {len(channels)} 个频道")
    print("✅ 分类完成。")

    print("\n➡️ 步骤 5/5: 生成最终文件...")
    total_saved = save_files(categorized_lists)

    print("\n--- 任务完成 ---")
    timeend = datetime.now()
    elapsed = timeend - timestart
    minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
    print(f"📊 总耗时: {minutes}分 {seconds}秒")
    print(f"📊 总计有效频道数: {total_saved}")

if __name__ == "__main__":
    main()

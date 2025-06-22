import streamlit as st
from user import main
import time
import os
import base64
from user_data_storage import read_credentials, write_credentials, storage_file

# 用户数据存储
class Credentials:
    def __init__(self, username, password, nickname, is_admin=False):
        self.username = username
        self.password = password
        self.nickname = nickname
        self.is_admin = is_admin
        
    def to_dict(self):
        return {
            'username': self.username,
            'password': self.password,
            'nickname': self.nickname,
            'is_admin': self.is_admin
        }

# 从文件读取用户数据
credentials = read_credentials(storage_file)

def write_credentials_to_file(username, password, nickname, is_admin=False):
    global credentials
    credentials[username] = Credentials(username, password, nickname, is_admin)
    write_credentials(storage_file, credentials)

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"读取文件错误: {str(e)}")
        return ""

def set_background():
    # 使用相对路径
    img_path = os.path.join("img", "login_background.jpg")
    bin_str = get_base64_of_bin_file(img_path)
    if not bin_str:
        return
        
    page_bg_img = '''
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("data:image/jpg;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        animation: fadeIn 1s ease-out;
    }
    [data-testid="stSidebar"] {
        background-color: transparent !important;
    }
    .stApp > header {
        background-color: transparent !important;
    }
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    @keyframes fadeOut {
        from {
            opacity: 1;
        }
        to {
            opacity: 0;
        }
    }
    .fade-out {
        animation: fadeOut 0.5s ease-out forwards;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# 设置背景图片
set_background()

# 添加自定义CSS样式
st.markdown("""
<style>
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 2rem;
    }
    .stForm {
        background-color: rgba(255, 255, 255, 0.02);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 400px;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.8s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        animation: fadeIn 0.8s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        width: 200px !important;
        margin: 0 auto !important;
        display: block !important;
        background: linear-gradient(45deg, rgba(64, 149, 255, 0.98), rgba(94, 114, 235, 0.98));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
        animation: slideUp 0.5s ease-out;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(64, 149, 255, 0.4);
        background: linear-gradient(45deg, rgba(94, 114, 235, 0.98), rgba(64, 149, 255, 0.98));
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.75rem;
        background-color: rgba(32, 33, 35, 0.98);
        color: #fff;
        backdrop-filter: blur(5px);
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: rgba(64, 149, 255, 0.6);
        box-shadow: 0 0 0 2px rgba(64, 149, 255, 0.2);
        background-color: rgba(32, 33, 35, 0.99);
    }
    .stTextInput>div>div>input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    .stTextInput>div>label {
        color: rgba(255, 255, 255, 0.9) !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        animation: fadeIn 0.5s ease-out;
        font-weight: 500;
    }
    .title {
        text-align: center;
        color: #ffffff;
        font-size: 3rem;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);
        letter-spacing: 2px;
        animation: fadeInDown 1s ease-out;
        padding: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        font-size: 1.3rem;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.5);
        letter-spacing: 1px;
        animation: fadeInUp 1s ease-out;
        font-weight: bold;
    }
    .mode-switch {
        text-align: center;
        margin: 1rem 0 0.5rem 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.5);
        animation: fadeIn 0.8s ease-out;
    }
    .success-message {
        text-align: center;
        color: #4ade80;
        font-weight: bold;
        animation: slideDown 0.5s ease-out;
        background-color: rgba(74, 222, 128, 0.1);
        padding: 0.75rem;
        border-radius: 8px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(74, 222, 128, 0.2);
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    .error-message {
        text-align: center;
        color: #f87171;
        font-weight: bold;
        animation: shake 0.5s ease-in;
        background-color: rgba(248, 113, 113, 0.1);
        padding: 0.75rem;
        border-radius: 8px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(248, 113, 113, 0.2);
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes slideRight {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    @keyframes slideOutUp {
        from {
            opacity: 1;
            transform: translateY(0);
        }
        to {
            opacity: 0;
            transform: translateY(-30px);
        }
    }
    .main-content {
        animation: fadeIn 0.8s ease-out;
    }
    .main-content.fade-out {
        animation: fadeOut 0.5s ease-out forwards;
    }
    .login-container {
        animation: fadeIn 0.8s ease-out;
    }
    .login-container.slide-out {
        animation: slideOutUp 0.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'admin' not in st.session_state:
    st.session_state.admin = False
if 'usname' not in st.session_state:
    st.session_state.usname = ""
if 'show_register' not in st.session_state:
    st.session_state.show_register = False

def login_page():
    container = st.container()
    with container:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><h1 class="title">🎬 泡泡Dragon电影助手</h1><p class="subtitle">带上爆米花，让 AI 为您开启专属的电影之旅 🍿</p></div>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("用户名", value="", placeholder="请输入用户名").strip()
            password = st.text_input("密码", value="", type="password", placeholder="请输入密码").strip()
            submit = st.form_submit_button("登录")
            
            if submit:
                if not username or not password:
                    st.markdown('<p class="error-message">用户名和密码不能为空！</p>', unsafe_allow_html=True)
                else:
                    user_cred = credentials.get(username)
                    if user_cred and user_cred.password == password:
                        st.markdown('<p class="success-message">登录成功！</p>', unsafe_allow_html=True)
                        st.markdown('<script>document.querySelector(".login-container").classList.add("slide-out");</script>', unsafe_allow_html=True)
                        st.session_state.logged_in = True
                        st.session_state.admin = user_cred.is_admin
                        st.session_state.usname = user_cred.nickname or username
                        time.sleep(0.5)  # 等待动画完成
                        st.experimental_rerun()
                    else:
                        st.markdown('<p class="error-message">用户名或密码错误，请重新输入。</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mode-switch">还没有账号？</div>', unsafe_allow_html=True)
    if st.button("立即注册", key="register_button"):
        st.session_state.show_register = True
        st.experimental_rerun()
    
    # 添加数据更新时间标注
    st.markdown('<div style="position: fixed; bottom: 10px; right: 10px; color: rgba(255, 255, 255, 0.5); font-size: 0.8rem;">最后一次数据更新于：2024年12月</div>', unsafe_allow_html=True)

def register_page():
    container = st.container()
    with container:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><h1 class="title">🎬 泡泡Dragon电影助手</h1><p class="subtitle">成为我们的一员，发现更多精彩电影 ✨</p></div>', unsafe_allow_html=True)
        
        with st.form("register_form"):
            new_username = st.text_input("设置用户名", value="", placeholder="请输入用户名").strip()
            new_nickname = st.text_input("设置昵称", value="", placeholder="请输入您的昵称").strip()
            new_password = st.text_input("设置密码", value="", type="password", placeholder="请输入密码").strip()
            confirm_password = st.text_input("确认密码", value="", type="password", placeholder="请再次输入密码").strip()
            is_admin = False
            register_submit = st.form_submit_button("注册")
            
            if register_submit:
                if not new_username or not new_password or not confirm_password:
                    st.markdown('<p class="error-message">用户名和密码不能为空！</p>', unsafe_allow_html=True)
                elif not new_nickname:
                    st.markdown('<p class="error-message">昵称不能为空！</p>', unsafe_allow_html=True)
                elif len(new_username) < 3:
                    st.markdown('<p class="error-message">用户名长度不能少于3个字符！</p>', unsafe_allow_html=True)
                elif len(new_password) < 6:
                    st.markdown('<p class="error-message">密码长度不能少于6个字符！</p>', unsafe_allow_html=True)
                elif new_password != confirm_password:
                    st.markdown('<p class="error-message">两次输入的密码不一致！</p>', unsafe_allow_html=True)
                elif new_username in credentials:
                    st.markdown('<p class="error-message">用户名已存在，请使用其他用户名。</p>', unsafe_allow_html=True)
                else:
                    write_credentials_to_file(new_username, new_password, new_nickname, is_admin)
                    st.markdown(f'<p class="success-message">用户 {new_nickname} 注册成功！请登录。</p>', unsafe_allow_html=True)
                    st.markdown('<script>document.querySelector(".login-container").classList.add("slide-out");</script>', unsafe_allow_html=True)
                    time.sleep(0.5)  # 等待动画完成
                    st.session_state.show_register = False
                    st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mode-switch">已有账号？</div>', unsafe_allow_html=True)
    if st.button("立即登录", key="login_button"):
        st.session_state.show_register = False
        st.experimental_rerun()

if __name__ == "__main__":
    if not st.session_state.logged_in:
        if st.session_state.show_register:
            register_page()
        else:
            login_page()
    else:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        main(st.session_state.admin, st.session_state.usname)
        st.markdown('</div>', unsafe_allow_html=True)

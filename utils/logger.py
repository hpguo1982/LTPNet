import logging
import os
from datetime import datetime
from registers import AIITModel


class AIITLogger(AIITModel):
    def __init__(self, name="MyLogger", log_dir="logs", level=logging.INFO):
        """
        :param name: logger 名称
        :param log_dir: 日志文件保存目录
        :param level: 日志等级
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # 防止重复输出

        # 格式化
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # 控制台输出
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

        # 文件输出
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)


if __name__ == "__main__":

    log = SimpleLogger(name="demo", log_dir="logs", level=logging.DEBUG)

    log.info("程序开始运行")
    log.debug("这是调试信息")
    log.warning("这是警告信息")
    log.error("这是错误信息")
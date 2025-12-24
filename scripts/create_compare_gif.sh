#!/bin/bash

# 创建compare图片GIF动图的脚本
# 将指定目录中带有'compare'的图片序列制作成GIF

set -e

# 定义颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 定义路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$PROJECT_ROOT/experiments/i3net/refined_i3net_finetune_test/visual_results/20251104_倪誉茹_CT481124_154804_1.0 x 1.0_HD_203"
OUTPUT_DIR="$PROJECT_ROOT/experiments/i3net/refined_i3net_finetune_test/visual_results"
OUTPUT_FILE="$OUTPUT_DIR/20251104_倪誉茹_CT481124_154804_compare_finetune.gif"

# 可选：从命令行参数指定每帧延迟
DURATION=${1:-100}  # 默认100ms

echo -e "${YELLOW}================================${NC}"
echo -e "${YELLOW}Compare 图片 GIF 生成脚本${NC}"
echo -e "${YELLOW}================================${NC}"
echo ""

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ 错误：输入目录不存在${NC}"
    echo "路径: $INPUT_DIR"
    exit 1
fi

echo -e "${GREEN}✓ 输入目录存在${NC}"
echo "  路径: $INPUT_DIR"
echo ""

# 检查compare图片数量
COMPARE_COUNT=$(ls "$INPUT_DIR" | grep -c "compare" || true)
if [ "$COMPARE_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ 错误：未找到任何compare图片${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 找到 $COMPARE_COUNT 张compare图片${NC}"
echo ""

# 检查Python和Pillow
echo "检查依赖..."
python3 -c "from PIL import Image" 2>/dev/null || {
    echo -e "${YELLOW}⚠ 未安装Pillow，正在安装...${NC}"
    pip3 install Pillow
}
echo -e "${GREEN}✓ 依赖检查完成${NC}"
echo ""

# 执行Python脚本
echo -e "${YELLOW}生成GIF...${NC}"
python3 "$SCRIPT_DIR/create_compare_gif.py" "$OUTPUT_FILE" "$DURATION"

echo ""
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo -e "${GREEN}✓ 生成成功！${NC}"
    echo "  输出文件: $OUTPUT_FILE"
    echo "  文件大小: $SIZE"
    echo "  每帧延迟: ${DURATION}ms"
else
    echo -e "${RED}❌ 生成失败！${NC}"
    exit 1
fi

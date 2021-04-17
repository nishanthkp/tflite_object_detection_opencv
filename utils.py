import cv2


Color = {
    'BLUE': (255, 0, 0),
    'LIME': (0, 255, 0),
    'RED': (0, 0, 255),
    'BLACK': (0, 0, 0),
    'WHITE': (255, 255, 255),
    'YELLOW': (0, 255, 255),
    'CYAN': (255, 255, 0),
    'MAGENTA': (255, 0, 255),
    'GRAY': (128, 128, 128),
    'MAROON': (0, 0, 128),
    'OLIVE': (0, 128, 128),
    'GREEN': (0, 128, 0),
    'PURPLE': (128, 0, 128),
    'TEAL': (128, 128, 0),
    'NAVY': (128, 0, 0)
}


def draw_text_with_background(img, text, org, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                              text_color=Color['BLACK'], thickness=1, line_type=cv2.LINE_AA, bottom_left_origin=False,
                              background_color=Color['WHITE']):
    text_dimensions, _ = cv2.getTextSize(text, font_face, font_scale, thickness)

    text_offset_x, text_offset_y = org
    text_width, text_height = text_dimensions

    cv2.rectangle(img, (text_offset_x, text_offset_y-text_height), (text_offset_x+text_width, text_offset_y),
                  background_color, cv2.FILLED)
    cv2.putText(img, text, org, font_face, font_scale, text_color, thickness, line_type, bottom_left_origin)

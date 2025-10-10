class InputData:
    uri_sources: list
    cam_ids: list
    extra_informations: list

    def __init__(self) -> None:
        self.uri_sources = []
        self.cam_ids = []
        self.extra_informations = []

    def add_source(self, uri_source: str, cam_id: str, extra_information: dict) -> None:
        self.uri_sources.append(uri_source)
        self.cam_ids.append(cam_id)
        self.extra_informations.append(extra_information)

    def get_src(self):
        return self.uri_sources

    def get_cams_id(self):
        return self.cam_ids

    def get_extra_informations(self):
        return self.extra_informations

    def clean_source(self) -> None:
        self.uri_sources.clear()
        self.cam_ids.clear()
        self.extra_informations.clear()

    def get_size(self):
        return len(self.uri_sources)

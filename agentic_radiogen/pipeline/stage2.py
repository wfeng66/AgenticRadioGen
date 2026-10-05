from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.schemas.contracts import GenomicMatrix, ImageBundle, OmicsBundle, RadiomicMatrix


@dataclass
class SpecialistOutputs:
    radiomics: RadiomicMatrix | None = None
    genomics: GenomicMatrix | None = None
    imaging_error: str | None = None
    genomics_error: str | None = None

    @property
    def patient_ids(self) -> list[str]:
        radio_ids = set(self.radiomics.patient_ids) if self.radiomics else set()
        geno_ids = set(self.genomics.patient_ids) if self.genomics else set()
        if radio_ids and geno_ids:
            return sorted(radio_ids & geno_ids)
        return sorted(radio_ids or geno_ids)


def extract_parallel(
    images: ImageBundle,
    omics: OmicsBundle,
    *,
    imaging: ImagingRadiomicsAgent | None = None,
    genomics: GenomicsAgent | None = None,
) -> SpecialistOutputs:
    """Run imaging and genomics independently. Neither agent is given the other's input."""
    imaging = imaging or ImagingRadiomicsAgent()
    genomics = genomics or GenomicsAgent()
    with ThreadPoolExecutor(max_workers=2) as pool:
        radio_future = pool.submit(imaging.extract, images)
        geno_future = pool.submit(genomics.extract, omics)
        radiomics, imaging_error = _settle(radio_future)
        genomics_matrix, genomics_error = _settle(geno_future)
    return SpecialistOutputs(
        radiomics=radiomics,
        genomics=genomics_matrix,
        imaging_error=imaging_error,
        genomics_error=genomics_error,
    )


def _settle(future) -> tuple[RadiomicMatrix | GenomicMatrix | None, str | None]:
    try:
        return future.result(), None
    except Exception as exc:
        return None, str(exc)

"""
Avatar Asset Validation Utility
Validates default and bundled avatar assets for the Avatar Streaming Service
"""

import os
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class AvatarAssetValidator:
    """Validates avatar assets in the assets directory"""
    
    def __init__(self, assets_path: str = None):
        self.assets_path = Path(assets_path) if assets_path else Path(__file__).parent
        self.validation_cache = {}
        
        # Expected asset specifications
        self.expected_assets = {
            'default_avatar.jpg': {
                'min_size': (256, 256),
                'max_size': (1024, 1024),
                'format': 'JPEG',
                'required': True
            },
            'persian_avatar.jpg': {
                'min_size': (256, 256),
                'max_size': (1024, 1024),
                'format': 'JPEG',
                'required': True
            }
        }
    
    def validate_all_assets(self) -> Dict[str, any]:
        """Validate all avatar assets"""
        results = {
            'overall_status': 'pass',
            'assets': {},
            'missing_assets': [],
            'issues': []
        }
        
        logger.info(f"Validating avatar assets in {self.assets_path}")
        
        # Check each expected asset
        for asset_name, specs in self.expected_assets.items():
            asset_path = self.assets_path / asset_name
            
            if not asset_path.exists():
                if specs['required']:
                    results['missing_assets'].append(asset_name)
                    results['overall_status'] = 'fail'
                    logger.error(f"Required asset missing: {asset_name}")
                continue
            
            # Validate individual asset
            asset_result = self.validate_asset(asset_path, specs)
            results['assets'][asset_name] = asset_result
            
            if not asset_result['valid']:
                results['overall_status'] = 'fail'
                results['issues'].extend([f"{asset_name}: {issue}" for issue in asset_result['issues']])
        
        # Check for unexpected files
        unexpected_files = self._find_unexpected_files()
        if unexpected_files:
            results['issues'].extend([f"Unexpected file: {f}" for f in unexpected_files])
        
        logger.info(f"Asset validation complete. Status: {results['overall_status']}")
        return results
    
    def validate_asset(self, asset_path: Path, specs: Dict) -> Dict[str, any]:
        """Validate individual asset file"""
        result = {
            'valid': True,
            'path': str(asset_path),
            'size_bytes': 0,
            'dimensions': None,
            'format': None,
            'issues': []
        }
        
        try:
            # Check file existence and basic properties
            if not asset_path.exists():
                result['valid'] = False
                result['issues'].append("File does not exist")
                return result
            
            result['size_bytes'] = asset_path.stat().st_size
            
            # Validate as image
            try:
                with Image.open(asset_path) as img:
                    result['dimensions'] = img.size
                    result['format'] = img.format
                    
                    # Check format
                    if result['format'] != specs['format']:
                        result['issues'].append(f"Expected format {specs['format']}, got {result['format']}")
                        result['valid'] = False
                    
                    # Check dimensions
                    width, height = result['dimensions']
                    min_w, min_h = specs['min_size']
                    max_w, max_h = specs['max_size']
                    
                    if width < min_w or height < min_h:
                        result['issues'].append(f"Image too small: {width}x{height}, minimum: {min_w}x{min_h}")
                        result['valid'] = False
                    
                    if width > max_w or height > max_h:
                        result['issues'].append(f"Image too large: {width}x{height}, maximum: {max_w}x{max_h}")
                        result['valid'] = False
                    
                    # Check if image is readable and valid
                    try:
                        img.verify()
                    except Exception as e:
                        result['issues'].append(f"Image verification failed: {e}")
                        result['valid'] = False
            
            except Exception as e:
                result['issues'].append(f"Failed to open as image: {e}")
                result['valid'] = False
        
        except Exception as e:
            result['issues'].append(f"Validation error: {e}")
            result['valid'] = False
        
        return result
    
    def validate_face_detectability(self, asset_path: Path) -> Dict[str, any]:
        """Check if faces can be detected in the avatar"""
        result = {
            'faces_detected': 0,
            'face_confidence': 0.0,
            'detectability_score': 0.0,
            'issues': []
        }
        
        try:
            # This would integrate with the face detection system
            # For now, we'll do basic validation
            
            with Image.open(asset_path) as img:
                # Convert to numpy array for analysis
                img_array = np.array(img)
                
                # Basic heuristics for face detectability
                # Check if image has reasonable contrast
                if img.mode == 'RGB':
                    gray = img.convert('L')
                    gray_array = np.array(gray)
                    
                    # Check contrast
                    contrast = gray_array.std()
                    if contrast < 20:
                        result['issues'].append("Low contrast image may affect face detection")
                    
                    # Check brightness
                    brightness = gray_array.mean()
                    if brightness < 50 or brightness > 200:
                        result['issues'].append("Poor lighting may affect face detection")
                    
                    # Basic face detection simulation
                    # In real implementation, this would use InsightFace
                    result['detectability_score'] = min(1.0, (contrast / 50.0) * (1.0 - abs(brightness - 128) / 128.0))
                    
                    if result['detectability_score'] > 0.7:
                        result['faces_detected'] = 1
                        result['face_confidence'] = result['detectability_score']
                    
        except Exception as e:
            result['issues'].append(f"Face detectability check failed: {e}")
        
        return result
    
    def generate_asset_checksums(self) -> Dict[str, str]:
        """Generate checksums for all valid assets"""
        checksums = {}
        
        for asset_name in self.expected_assets.keys():
            asset_path = self.assets_path / asset_name
            if asset_path.exists():
                try:
                    with open(asset_path, 'rb') as f:
                        content = f.read()
                        checksum = hashlib.sha256(content).hexdigest()
                        checksums[asset_name] = checksum
                except Exception as e:
                    logger.error(f"Failed to generate checksum for {asset_name}: {e}")
        
        return checksums
    
    def create_asset_manifest(self) -> Dict[str, any]:
        """Create a manifest of all assets"""
        manifest = {
            'version': '1.0',
            'generated_at': None,
            'assets': {},
            'checksums': self.generate_asset_checksums()
        }
        
        # Validate and add asset information
        for asset_name, specs in self.expected_assets.items():
            asset_path = self.assets_path / asset_name
            if asset_path.exists():
                validation_result = self.validate_asset(asset_path, specs)
                face_result = self.validate_face_detectability(asset_path)
                
                manifest['assets'][asset_name] = {
                    'validation': validation_result,
                    'face_detectability': face_result,
                    'specification': specs
                }
        
        return manifest
    
    def _find_unexpected_files(self) -> List[str]:
        """Find files that aren't in the expected assets list"""
        unexpected = []
        
        try:
            for item in self.assets_path.iterdir():
                if item.is_file() and item.name not in self.expected_assets:
                    # Skip common non-asset files
                    if not item.name.startswith('.') and item.name != '__pycache__':
                        unexpected.append(item.name)
        except Exception as e:
            logger.error(f"Error scanning for unexpected files: {e}")
        
        return unexpected
    
    def repair_assets(self, create_missing: bool = False) -> Dict[str, any]:
        """Attempt to repair or recreate missing/damaged assets"""
        repair_results = {
            'repairs_attempted': [],
            'repairs_successful': [],
            'repairs_failed': [],
            'warnings': []
        }
        
        validation_results = self.validate_all_assets()
        
        # Handle missing assets
        for missing_asset in validation_results['missing_assets']:
            if create_missing:
                repair_results['repairs_attempted'].append(f"Create missing: {missing_asset}")
                
                try:
                    success = self._create_placeholder_asset(missing_asset)
                    if success:
                        repair_results['repairs_successful'].append(missing_asset)
                    else:
                        repair_results['repairs_failed'].append(missing_asset)
                except Exception as e:
                    repair_results['repairs_failed'].append(f"{missing_asset}: {e}")
        
        # Handle damaged assets
        for asset_name, asset_result in validation_results['assets'].items():
            if not asset_result['valid']:
                repair_results['warnings'].append(f"Asset {asset_name} has issues: {asset_result['issues']}")
        
        return repair_results
    
    def _create_placeholder_asset(self, asset_name: str) -> bool:
        """Create a placeholder asset file"""
        try:
            specs = self.expected_assets[asset_name]
            asset_path = self.assets_path / asset_name
            
            # Create a simple placeholder image
            width, height = specs['min_size']
            
            # Create a colored placeholder image
            if 'persian' in asset_name.lower():
                # Persian-themed colors
                color = (70, 130, 180)  # Steel blue
            else:
                # Default neutral color
                color = (128, 128, 128)  # Gray
            
            placeholder_img = Image.new('RGB', (width, height), color)
            
            # Save the placeholder
            placeholder_img.save(asset_path, format=specs['format'])
            
            logger.info(f"Created placeholder asset: {asset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create placeholder for {asset_name}: {e}")
            return False

def main():
    """Main function for running asset validation"""
    validator = AvatarAssetValidator()
    
    print("Avatar Asset Validation")
    print("=" * 50)
    
    # Run validation
    results = validator.validate_all_assets()
    
    print(f"Overall Status: {results['overall_status'].upper()}")
    print()
    
    # Show asset details
    if results['assets']:
        print("Asset Details:")
        for asset_name, asset_info in results['assets'].items():
            status = "✓ PASS" if asset_info['valid'] else "✗ FAIL"
            print(f"  {asset_name}: {status}")
            if asset_info['dimensions']:
                print(f"    Dimensions: {asset_info['dimensions'][0]}x{asset_info['dimensions'][1]}")
                print(f"    Format: {asset_info['format']}")
                print(f"    Size: {asset_info['size_bytes']} bytes")
            if asset_info['issues']:
                for issue in asset_info['issues']:
                    print(f"    Issue: {issue}")
        print()
    
    # Show missing assets
    if results['missing_assets']:
        print("Missing Assets:")
        for asset in results['missing_assets']:
            print(f"  ✗ {asset}")
        print()
    
    # Show overall issues
    if results['issues']:
        print("Issues:")
        for issue in results['issues']:
            print(f"  • {issue}")
        print()
    
    # Generate manifest
    manifest = validator.create_asset_manifest()
    manifest_path = validator.assets_path / 'asset_manifest.json'
    
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"Asset manifest saved to: {manifest_path}")
    except Exception as e:
        print(f"Failed to save manifest: {e}")
    
    return results['overall_status'] == 'pass'

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 